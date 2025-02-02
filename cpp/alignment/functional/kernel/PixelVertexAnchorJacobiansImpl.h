//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/27/23.
//  Copyright (c) 2023 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>

// local includes
#include "alignment/functional/kernel/PixelVertexAnchorJacobians.h"
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/linalg/KroneckerTensorProduct.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/Atomics.h"
#include "geometry/functional/kernel/Defines.h"
#include "alignment/functional/FaceNodeAnchors.h"

#define MAX_PIXELS_PER_NODE 4000

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::alignment::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType, bool TUseTukeyPenalty, IterationMode TIterationMode>
void PixelVertexAnchorJacobiansAndNodeAssociations_Generic(
		open3d::core::Tensor& pixel_node_jacobians,
		open3d::core::Tensor& pixel_node_jacobian_counts,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		const open3d::core::Tensor& face_node_anchor_counts,
		int64_t node_count,
		float tukey_penalty_cutoff
) {

	// === dimension, type, and device tensor checks; counts ===
	int64_t image_height = rasterized_vertex_position_jacobians.GetShape(0);
	int64_t image_width = rasterized_vertex_position_jacobians.GetShape(1);
	int64_t pixel_count = image_width * image_height;

	o3c::Device device = residual_mask.GetDevice();

	o3c::AssertTensorShape(rasterized_vertex_position_jacobians, { image_height, image_width, 3, 9 });
	o3c::AssertTensorDevice(rasterized_vertex_position_jacobians, device);
	o3c::AssertTensorDtype(rasterized_vertex_position_jacobians, o3c::Float32);

	o3c::AssertTensorShape(rasterized_vertex_normal_jacobians, { image_height, image_width, 3, 10 });
	o3c::AssertTensorDevice(rasterized_vertex_normal_jacobians, device);
	o3c::AssertTensorDtype(rasterized_vertex_normal_jacobians, o3c::Float32);

	int64_t vertex_count = warped_vertex_position_jacobians.GetShape(0);
	int64_t anchor_per_vertex_count = warped_vertex_position_jacobians.GetShape(1);
	if (anchor_per_vertex_count > MAX_ANCHOR_COUNT) {
		utility::LogError("Supplied anchor count per vertex, {}, is greater than the allowed maximum, {}.",
		                  anchor_per_vertex_count, MAX_ANCHOR_COUNT);
	}

	if (TIterationMode == IterationMode::TRANSLATION_ONLY) {
		// in the translation-only case, we expect vertex anchor weights to be passed in directly
		o3c::AssertTensorShape(warped_vertex_position_jacobians, { vertex_count, anchor_per_vertex_count });
	} else {
		o3c::AssertTensorShape(warped_vertex_position_jacobians, { vertex_count, anchor_per_vertex_count, 4 });
	}

	o3c::AssertTensorDevice(warped_vertex_position_jacobians, device);
	o3c::AssertTensorDtype(warped_vertex_position_jacobians, o3c::Float32);

	if (TIterationMode == IterationMode::ALL ||
	    TIterationMode == IterationMode::ROTATION_ONLY) {
		if (!warped_vertex_normal_jacobians.has_value()) {
			utility::LogError("warped_vertex_normal_jacobians argument has to be a tensor for a call to RasterizedSurfaceJacobians with"
			                  "TIterationMode template argument set to ALL or ROTATION_ONLY, which it is not.");
		}

		// warped vertex normal jacobians are irrelevant for translation-only case (vertex normals don't change with only translation)
		o3c::AssertTensorShape(warped_vertex_normal_jacobians.value().get(), { vertex_count, anchor_per_vertex_count, 3 });
		o3c::AssertTensorDevice(warped_vertex_normal_jacobians.value().get(), device);
		o3c::AssertTensorDtype(warped_vertex_normal_jacobians.value().get(), o3c::Float32);
	}

	o3c::AssertTensorShape(point_map_vectors, { pixel_count, 3 });
	o3c::AssertTensorDevice(point_map_vectors, device);
	o3c::AssertTensorDtype(point_map_vectors, o3c::Float32);

	o3c::AssertTensorShape(rasterized_normals, { pixel_count, 3 });
	o3c::AssertTensorDevice(rasterized_normals, device);
	o3c::AssertTensorDtype(rasterized_normals, o3c::Float32);

	o3c::AssertTensorShape(residual_mask, { pixel_count });
	o3c::AssertTensorDtype(residual_mask, o3c::Bool);

	int64_t faces_per_pixel = pixel_faces.GetShape(2);
	o3c::AssertTensorShape(pixel_faces, { image_height, image_width, faces_per_pixel });
	o3c::AssertTensorDevice(pixel_faces, device);
	o3c::AssertTensorDtype(pixel_faces, o3c::Int64);

	int64_t face_count = face_node_anchor_counts.GetShape(0);
	o3c::AssertTensorShape(face_node_anchor_counts, { face_count });
	o3c::AssertTensorDevice(face_node_anchor_counts, device);
	o3c::AssertTensorDtype(face_node_anchor_counts, o3c::Int32);

	if(face_node_anchors->GetDevice() != device){
		utility::LogError("face_node_anchors need to have the same device as all other argument tensors (which is {}), got {} instead.",
						  device.ToString(), face_node_anchors->GetDevice().ToString());
	}

	// === initialize output matrices ===
	node_pixel_jacobian_indices = o3c::Tensor({node_count, MAX_PIXELS_PER_NODE}, o3c::Int32, device);

	node_pixel_jacobian_indices.Fill(-1);

	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();

	core::AtomicCounterArray<TDeviceType> node_pixel_counters(node_count);

	// each anchor node controls from 1 to 3 vertices per face intersecting with the pixel ray:
	// these jacobians are aggregated on a per-node basis
	// thus, count of jacobians per vertex <= [count of anchor nodes per vertex] * 3
	// [pixel count] x ([count of anchor nodes per vertex] * 3) x [6 max. values per node , i.e. 3 rotation angles, 3 translation coordinates]
	int jacobian_stride;
	if (TIterationMode == IterationMode::ALL) {
		jacobian_stride = 6;
	} else {
		jacobian_stride = 3;
	}
	const int max_face_anchor_count = 3 * static_cast<int>(anchor_per_vertex_count);
	pixel_node_jacobians = o3c::Tensor::Zeros({pixel_count, max_face_anchor_count, jacobian_stride}, o3c::Float32, device);
	auto pixel_node_jacobians_data = pixel_node_jacobians.GetDataPtr<float>();
	pixel_node_jacobian_counts = o3c::Tensor::Zeros({pixel_count}, o3c::Int32, device);
	auto pixel_node_jacobian_counts_data = pixel_node_jacobian_counts.GetDataPtr<int32_t>();

	// === get access to raw input data
	auto residual_mask_data = static_cast<const unsigned char*>(residual_mask.GetDataPtr());
	auto point_map_vector_data = point_map_vectors.GetDataPtr<float>();
	auto rasterized_normal_data = rasterized_normals.GetDataPtr<float>();

	auto rasterized_vertex_position_jacobian_data = rasterized_vertex_position_jacobians.GetDataPtr<float>();
	auto rasterized_vertex_normal_jacobian_data = rasterized_vertex_normal_jacobians.GetDataPtr<float>();

	auto warped_vertex_position_jacobian_data = warped_vertex_position_jacobians.GetDataPtr<float>();
	const float* warped_vertex_normal_jacobian_data = nullptr;

	if (TIterationMode == IterationMode::ALL ||
	    TIterationMode == IterationMode::ROTATION_ONLY) {
		warped_vertex_normal_jacobian_data = warped_vertex_normal_jacobians.value().get().GetDataPtr<float>();
	}

	auto pixel_face_data = pixel_faces.template GetDataPtr<int64_t>();
	auto face_node_anchor_data = reinterpret_cast<FaceNodeAnchors*>(face_node_anchors->GetDataPtr());
	auto face_node_anchor_count_data = face_node_anchor_counts.GetDataPtr<int32_t>();



	// === loop over all pixels & compute
	o3c::ParallelFor(
			device, pixel_count,
			//TODO first do this on a per-face level (remove the pixel_point_map_vector and pixel_rasterized_normal), precompute per-face jacobians first
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t pixel_index) {
				if (!residual_mask_data[pixel_index]) {
					return;
				}
				auto v_image = static_cast<int>(pixel_index / image_width);
				auto u_image = static_cast<int>(pixel_index % image_width);

				// ==== VARIABLE NAME LEGEND ====
				// r stands for "residuals"
				// subscript l refers to pixel l
				// wl and nl stand for rasterized warped vertex positions and normals, respectively
				// V and N stand for warped vertex positions and normals, respectively

				Eigen::Map<const Eigen::RowVector3f> wl_minus_ol(point_map_vector_data + pixel_index * 3);
				Eigen::Map<const Eigen::RowVector3f> nl(rasterized_normal_data + pixel_index * 3);
				//TODO: potential optimization: use templated call to rest of code instead to avoid copying to dr_dnl, dr_dwl
				Eigen::RowVector3f dr_dnl, dr_dwl;

				if (TUseTukeyPenalty) {
					// recompute r, i.e. residual without applying the Tukey psi / penalty function
					float r = nl.dot(wl_minus_ol);

					if (fabsf(r) > tukey_penalty_cutoff) {
						return;
					}
					// ∂Tukey(r)/∂r = psi(r) = r * (1 - (r/c)^2)^2
					float quotient_r_and_c = r / tukey_penalty_cutoff;
					float psi_r = 1 - quotient_r_and_c * quotient_r_and_c;
					psi_r = r * psi_r * psi_r;


					// ∂Tukey(r)/∂nl = psi(r) * ∂nl(wl-ol)/∂nl = psi(r) * (wl-ol)^T
					dr_dnl = psi_r * wl_minus_ol;

					// ∂Tukey(r)/∂wl = psi(r) * ∂nl(wl-ol)/∂wl = nl^T
					dr_dwl = psi_r * nl;

					// note the switch above in meaning of variable "r": r'' = psi(r)
				} else {
					// ∂r/∂nl = ∂nl(wl-ol)/∂nl = (wl-ol)^T
					dr_dnl = wl_minus_ol;
					// ∂r/∂wl = ∂nl(wl-ol)/∂wl = nl^T
					dr_dwl = nl;
				}

				Eigen::Map<const core::kernel::Matrix3x9f> dwl_dV
						(rasterized_vertex_position_jacobian_data + pixel_index * (3 * 9));
				Eigen::Map<const core::kernel::Matrix3x9f> dnl_dV
						(rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10));

				// [1 x 3] * [3 x 9] = [1 x 9]
				auto dr_dwl_x_dwl_dV = dr_dwl * dwl_dV;
				// [1 x 3] * [3 x 9] = [1 x 9]
				auto dr_dnl_x_dnl_dV = dr_dnl * dnl_dV;
				// [1 x 6] * [6 x 9] = [1 x 9]
				auto dr_dV = dr_dwl_x_dwl_dV + dr_dnl_x_dnl_dV;

				Eigen::Matrix<float, 1, 9, Eigen::RowMajor> dr_dN;

				if (TIterationMode == IterationMode::ALL ||
				    TIterationMode == IterationMode::ROTATION_ONLY) {
					Eigen::Map<const Eigen::RowVector3f> pixel_barycentric_coordinates
							(rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10) + (3 * 9));
					//TODO: if/when Eigen::kroneckerProduct gets fixed, use the below line instead of custom function
					// auto dr_dN =
					// 		dr_dnl *
					// 		Eigen::kroneckerProduct(pixel_barycentric_coordinates, core::kernel::Matrix3f::Identity());
					Eigen::Matrix<float, 3, 9, Eigen::RowMajor> dnl_dN;
					nnrt::core::linalg::kernel::ComputeKroneckerProduct(dnl_dN, pixel_barycentric_coordinates,
					                                                    core::kernel::Matrix3f::Identity());

					// dr_dN = dr_dwl * dwl_dN + dr_dnl * dnl_dN = 0 + dr_dnl  * dnl_dN
					//                                                 [1 x 3] * [3 x 9] = [1 x 9]
					dr_dN = dr_dnl * dnl_dN;
				}

				auto i_face = pixel_face_data[(v_image * image_width * faces_per_pixel) + (u_image * faces_per_pixel)];

				int pixel_jacobian_list_address = static_cast<int>(pixel_index) * (static_cast<int>(anchor_per_vertex_count) * 3 * jacobian_stride);
				auto pixel_node_jacobian_data = pixel_node_jacobians_data + pixel_jacobian_list_address;

				FaceNodeAnchors* face_anchor_data = face_node_anchor_data + i_face * (3 * MAX_ANCHOR_COUNT);
				int32_t face_anchor_count = face_node_anchor_count_data[i_face];

				pixel_node_jacobian_counts_data[pixel_index] = face_anchor_count;

				// traverse all anchors associated with current face, compute their jacobians for current pixel
				for (int i_face_anchor = 0; i_face_anchor < face_anchor_count; i_face_anchor++) {
					auto& face_anchor = face_anchor_data[i_face_anchor];
					int i_node = face_anchor.node_index;

					int jacobian_local_address = i_face_anchor * jacobian_stride;

					float* pixeL_vertex_anchor_rotation_jacobian_data = nullptr;
					float* pixel_vertex_anchor_translation_jacobian_data = nullptr;

					switch (TIterationMode) {
						case IterationMode::ALL: pixeL_vertex_anchor_rotation_jacobian_data = pixel_node_jacobian_data + jacobian_local_address;
							pixel_vertex_anchor_translation_jacobian_data = pixel_node_jacobian_data + jacobian_local_address + 3;
							break;
						case IterationMode::ROTATION_ONLY:
							pixeL_vertex_anchor_rotation_jacobian_data = pixel_node_jacobian_data + jacobian_local_address;
							break;
						case IterationMode::TRANSLATION_ONLY:
							pixel_vertex_anchor_translation_jacobian_data = pixel_node_jacobian_data + jacobian_local_address;
							break;
					}

					for (int i_face_vertex = 0; i_face_vertex < 3; i_face_vertex++) {
						int i_vertex = face_anchor.vertices[i_face_vertex];
						if (i_vertex == -1) continue;
						int i_vertex_anchor = face_anchor.vertex_anchor_indices[i_face_vertex];

						// used to compute warped vertex position Jacobian w.r.t. node translation, weight * I_3x3
						float stored_anchor_weight;
						if (TIterationMode == IterationMode::TRANSLATION_ONLY) {
							// in translation-only mode, we expect the vertex weights to be passed in directly
							stored_anchor_weight = warped_vertex_position_jacobian_data[i_vertex * anchor_per_vertex_count + i_vertex_anchor];
						} else {
							stored_anchor_weight = warped_vertex_position_jacobian_data[
									(i_vertex * anchor_per_vertex_count * 4) +
									(i_vertex_anchor * 4) + 3
							];
						}


						// [1x3]
						auto dr_dv = dr_dV.block<1, 3>(0, i_face_vertex * 3);

						if (TIterationMode == IterationMode::ALL ||
						    TIterationMode == IterationMode::TRANSLATION_ONLY) {
							Eigen::Map<Eigen::RowVector3<float>>
									pixel_vertex_anchor_translation_jacobian(pixel_vertex_anchor_translation_jacobian_data);
							// [1x3] = ([1x3] * [3x3]) + ([1x3] * [3x3])
							pixel_vertex_anchor_translation_jacobian += dr_dv * (Eigen::Matrix3f::Identity() * stored_anchor_weight);
						}

						if (TIterationMode == IterationMode::ALL ||
						    TIterationMode == IterationMode::ROTATION_ONLY) {
							// [1x3]
							auto dr_dn = dr_dN.block<1, 3>(0, i_face_vertex * 3);

							// [3x3] warped vertex position Jacobian w.r.t. node rotation
							const Eigen::SkewSymmetricMatrix3<float> dv_drotation(
									Eigen::Map<const Eigen::Vector3f>(
											warped_vertex_position_jacobian_data +
											(i_vertex * anchor_per_vertex_count * 4) + // vertex index * stride
											(i_vertex_anchor * 4) // index of the anchor for this vertex * stride
									)
							);

							// [3x3] warped vertex normal Jacobian w.r.t. node rotation
							const Eigen::SkewSymmetricMatrix3<float> dn_drotation(
									Eigen::Map<const Eigen::Vector3f>(
											warped_vertex_normal_jacobian_data +
											(i_vertex * anchor_per_vertex_count * 3) +
											(i_vertex_anchor * 3)
									)
							);
							Eigen::Map<Eigen::RowVector3<float>> pixel_vertex_anchor_rotation_jacobian(pixeL_vertex_anchor_rotation_jacobian_data);
							// [1x3] = ([1x3] * [3x3]) + ([1x3] * [3x3])
							pixel_vertex_anchor_rotation_jacobian += (dr_dv * dv_drotation) + (dr_dn * dn_drotation);
						}
					}

					// accumulate addresses of jacobians for each node
					int i_node_jacobian = node_pixel_counters.FetchAdd(i_node, 1);
					// safety check
					if (i_node_jacobian > MAX_PIXELS_PER_NODE) {
						printf("Warning: number of pixels affected by node %i exceeds allowed maximum, %i. "
						       "Result may be incomplete. Either the voxel size (or triangle size) is simply too "
						       "large, or the surface is too close to the camera.\n", i_node, MAX_PIXELS_PER_NODE);
						node_pixel_counters.FetchSub(i_node, 1);
					} else {
						node_pixel_jacobian_index_data[i_node * MAX_PIXELS_PER_NODE + i_node_jacobian] =
								pixel_jacobian_list_address + jacobian_local_address;
					}
					face_anchor = face_anchor_data[i_face_anchor];
				}
			}

	);
	node_pixel_jacobian_counts = node_pixel_counters.AsTensor(true);
}


template<open3d::core::Device::DeviceType TDeviceType>
void PixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_jacobians,
		open3d::core::Tensor& pixel_jacobian_counts,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		const open3d::core::Tensor& face_node_anchor_counts,
		int64_t node_count,
		IterationMode mode,
		bool use_tukey_penalty,
		float tukey_penalty_cutoff
) {

#define NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS \
    pixel_jacobians, pixel_jacobian_counts, node_pixel_jacobian_indices, node_pixel_jacobian_counts, \
    rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,                        \
    warped_vertex_position_jacobians, warped_vertex_normal_jacobians,                                \
    point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_node_anchors,            \
    face_node_anchor_counts, node_count, tukey_penalty_cutoff
	if (use_tukey_penalty) {
		switch (mode) {
			case ALL:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, true, IterationMode::ALL>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
			case TRANSLATION_ONLY:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, true, IterationMode::TRANSLATION_ONLY>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
			case ROTATION_ONLY:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, true, IterationMode::ROTATION_ONLY>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
		}
	} else {
		switch (mode) {
			case ALL:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, false, IterationMode::ALL>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
			case TRANSLATION_ONLY:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, false, IterationMode::TRANSLATION_ONLY>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
			case ROTATION_ONLY:
				PixelVertexAnchorJacobiansAndNodeAssociations_Generic<TDeviceType, false, IterationMode::ROTATION_ONLY>(
						NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
				);
				break;
		}
	}
#undef NNRT_PIXEL_VERTEX_ANCHOR_JACOBIAN_ARGS
}

} // namespace nnrt::alignment::functional::kernel