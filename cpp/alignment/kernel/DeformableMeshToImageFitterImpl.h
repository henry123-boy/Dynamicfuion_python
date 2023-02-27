//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/10/23.
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
#include <Eigen/Dense>


// local includes
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/linalg/KroneckerTensorProduct.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/Atomics.h"
#include "geometry/functional/kernel/Defines.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

#define MAX_PIXELS_PER_NODE 4000


namespace nnrt::alignment::kernel {

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Swap(TScalar* a, TScalar* b) {
	TScalar tmp = *a;
	*a = *b;
	*b = tmp;
}

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Heapify(TScalar* array, int length, int root) {
	int largest = root;
	int l = 2 * root + 1;
	int r = 2 * root + 2;

	if (l < length && array[l] > array[largest]) {
		largest = l;
	}
	if (r < length && array[r] > array[largest]) {
		largest = r;
	}
	if (largest != root) {
		Swap<TScalar>(&array[root], &array[largest]);
		Heapify<TScalar>(array, length, largest);
	}
}

template<typename TScalar>
NNRT_DEVICE_WHEN_CUDACC void HeapSort(TScalar* array, int length) {
	for (int i = length / 2 - 1; i >= 0; i--) Heapify(array, length, i);

	for (int i = length - 1; i > 0; i--) {
		Swap<TScalar>(&array[0], &array[i]);
		Heapify<TScalar>(array, i, 0);
	}
}

template<open3d::core::Device::DeviceType TDeviceType>
void ComputePixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_node_jacobians,
		open3d::core::Tensor& pixel_node_jacobian_counts,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		const open3d::core::Tensor& warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& face_vertices,
		const open3d::core::Tensor& vertex_anchors,
		int64_t node_count
) {
	// === dimension, type, and device tensor checks ===
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
	int64_t anchor_count_per_vertex = warped_vertex_position_jacobians.GetShape(1);
	if (anchor_count_per_vertex > MAX_ANCHOR_COUNT) {
		utility::LogError("Supplied anchor count per vertex, {}, is greater than the allowed maximum, {}.",
		                  anchor_count_per_vertex, MAX_ANCHOR_COUNT);
	}

	o3c::AssertTensorShape(warped_vertex_position_jacobians, { vertex_count, anchor_count_per_vertex, 4 });
	o3c::AssertTensorDevice(warped_vertex_position_jacobians, device);
	o3c::AssertTensorDtype(warped_vertex_position_jacobians, o3c::Float32);

	o3c::AssertTensorShape(warped_vertex_normal_jacobians, { vertex_count, anchor_count_per_vertex, 3 });
	o3c::AssertTensorDevice(warped_vertex_normal_jacobians, device);
	o3c::AssertTensorDtype(warped_vertex_normal_jacobians, o3c::Float32);

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

	o3c::AssertTensorShape(face_vertices, { utility::nullopt, 3 });
	o3c::AssertTensorDevice(face_vertices, device);
	o3c::AssertTensorDtype(face_vertices, o3c::Int64);

	o3c::AssertTensorShape(vertex_anchors, { vertex_count, anchor_count_per_vertex });
	o3c::AssertTensorDevice(vertex_anchors, device);
	o3c::AssertTensorDtype(vertex_anchors, o3c::Int32);

	// === initialize output matrices ===
	node_pixel_jacobian_indices = o3c::Tensor({node_count, MAX_PIXELS_PER_NODE}, o3c::Int32);
	node_pixel_jacobian_indices.Fill(-1);
	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();

	core::AtomicCounterArray<TDeviceType> node_pixel_counters(node_count);

	// each anchor node controls from 1 to 3 vertices per face intersecting with the pixel ray:
	// these jacobians are aggregated on a per-node basis
	// thus, count of jacobians per vertex <= [count of anchor nodes per vertex] * 3
	// [pixel count] x ([count of anchor nodes per vertex] * 3) x [6 values per node , i.e. 3 rotation angles, 3 translation coordinates]
	pixel_node_jacobians = o3c::Tensor::Zeros({pixel_count, anchor_count_per_vertex * 3, 6}, o3c::Float32);
	auto pixel_node_jacobians_data = pixel_node_jacobians.GetDataPtr<float>();
	pixel_node_jacobian_counts = o3c::Tensor::Zeros({pixel_count}, o3c::Int32);
	auto pixel_node_jacobian_counts_data = pixel_node_jacobian_counts.GetDataPtr<int32_t>();


	// === get access to raw input data
	auto residual_mask_data = static_cast<const unsigned char*>(residual_mask.GetDataPtr());
	auto point_map_vector_data = point_map_vectors.GetDataPtr<float>();
	auto rasterized_normal_data = rasterized_normals.GetDataPtr<float>();

	auto rasterized_vertex_position_jacobian_data = rasterized_vertex_position_jacobians.GetDataPtr<float>();
	auto rasterized_vertex_normal_jacobian_data = rasterized_vertex_normal_jacobians.GetDataPtr<float>();

	auto warped_vertex_position_jacobian_data = warped_vertex_position_jacobians.GetDataPtr<float>();
	auto warped_vertex_normal_jacobian_data = warped_vertex_normal_jacobians.GetDataPtr<float>();

	auto pixel_face_data = pixel_faces.template GetDataPtr<int64_t>();
	auto triangle_index_data = face_vertices.GetDataPtr<int64_t>();
	auto vertex_anchor_data = vertex_anchors.GetDataPtr<int32_t>();

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
				Eigen::Map<const Eigen::RowVector3f> pixel_point_map_vector(point_map_vector_data + pixel_index * 3);
				Eigen::Map<const Eigen::RowVector3f> pixel_rasterized_normal(rasterized_normal_data + pixel_index * 3);

				Eigen::Map<const core::kernel::Matrix3x9f> pixel_vertex_position_jacobian
						(rasterized_vertex_position_jacobian_data + pixel_index * (3 * 9));
				Eigen::Map<const core::kernel::Matrix3x9f> pixel_rasterized_normal_jacobian
						(rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10));
				Eigen::Map<const Eigen::RowVector3f> pixel_barycentric_coordinates
						(rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10) + (3 * 9));
				// r stands for "residuals"
				// wl and nl stand for rasterized vertex positions and normals, respectively
				// V and N stand for warped vertex positions and normals, respectively
				// [1 x 3] * [3 x 9] = [1 x 9]
				auto dr_dwl_x_dwl_dV = pixel_rasterized_normal * pixel_vertex_position_jacobian;
				// [1 x 3] * [3 x 9] = [1 x 9]
				auto dr_dnl_x_dnl_dV = pixel_point_map_vector * pixel_rasterized_normal_jacobian;
				// [1 x 6] * [6 x 9] = [1 x 9]
				auto dr_dV = dr_dwl_x_dwl_dV + dr_dnl_x_dnl_dV;
				// dr_dN = dr_dl * dl_dV + dr_dn * dl_dN = 0 + dr_dn   * dl_dN
				//                                             [1 x 3] * [3 x 9] = [1 x 9]
				auto dr_dN =
						pixel_point_map_vector *
						Eigen::kroneckerProduct(pixel_barycentric_coordinates, core::kernel::Matrix3f::Identity());

				//__DEBUG
				// auto dr_dwl_x_dwl_dV_c = dr_dwl_x_dwl_dV.eval();
				// auto dr_dnl_x_dnl_dV_c = dr_dnl_x_dnl_dV.eval();
				// auto dr_dV_c = dr_dV.eval();
				// auto dr_dN_c = dr_dN.eval();

				auto i_face = pixel_face_data[(v_image * image_width * faces_per_pixel) + (u_image * faces_per_pixel)];
				Eigen::Map<const Eigen::RowVector3<int64_t>> vertex_indices(triangle_index_data + i_face * 3);
				int pixel_jacobian_list_address = static_cast<int>(pixel_index) * (static_cast<int>(anchor_count_per_vertex) * 3 * 6);
				auto pixel_node_jacobian_data = pixel_node_jacobians_data + pixel_jacobian_list_address;

				const int max_face_anchor_count = 3 * static_cast<int>(anchor_count_per_vertex);
				struct {
					int node_index = -1;
					int64_t vertices[3] = {-1, -1, -1};
					int vertex_anchor_indices[3] = {-1, -1, -1};
				} face_anchor_data[3 * MAX_ANCHOR_COUNT];

				// group vertices by anchor node
				//TODO: this simple per-face reindexing operation needs only to be done on a per-face basis. It is
				// probably much more efficient to outsource it to a separate kernel instead of rearranging things here
				// for every pixel.
				int32_t pixel_node_jacobian_count = 0;
				for (int i_face_vertex = 0; i_face_vertex < 3; i_face_vertex++) {
					int64_t i_vertex = vertex_indices(i_face_vertex);
					for (int i_vertex_anchor = 0; i_vertex_anchor < anchor_count_per_vertex; i_vertex_anchor++) {
						auto i_node = vertex_anchor_data[i_vertex * anchor_count_per_vertex + i_vertex_anchor];
						if (i_node == -1) {
							// -1 is the sentinel value for anchors
							continue;
						}
						int i_face_anchor = 0;
						int inspected_anchor_node_index = face_anchor_data[i_face_anchor].node_index;
						// find (if any) matching node among anchors of previous vertices
						while (
								inspected_anchor_node_index != i_node && // found match in previous matches
								inspected_anchor_node_index != -1 && // end of filled list reached
								i_face_anchor + 1 < max_face_anchor_count// end of list reached)
								) {
							i_face_anchor++;
							inspected_anchor_node_index = face_anchor_data[i_face_anchor].node_index;
						}

						auto& face_anchor = face_anchor_data[i_face_anchor];
						if (inspected_anchor_node_index != i_node) { // unique node found
							pixel_node_jacobian_count++; // tally per pixel
							face_anchor.node_index = i_node;
						}
						face_anchor.vertices[i_face_vertex] = i_vertex;
						face_anchor.vertex_anchor_indices[i_face_vertex] = i_vertex_anchor;
					}
				}
				pixel_node_jacobian_counts_data[pixel_index] = pixel_node_jacobian_count;

				// traverse all anchors associated with current face, compute their jacobians for current pixel
				for (int i_face_anchor = 0; i_face_anchor < pixel_node_jacobian_count; i_face_anchor++) {
					auto& face_anchor = face_anchor_data[i_face_anchor];
					int i_node = face_anchor.node_index;

					int jacobian_local_address = i_face_anchor * 6;

					Eigen::Map<Eigen::RowVector3<float>>
							pixel_vertex_anchor_rotation_jacobian
							(pixel_node_jacobian_data + jacobian_local_address);
					Eigen::Map<Eigen::RowVector3<float>>
							pixel_vertex_anchor_translation_jacobian
							(pixel_node_jacobian_data + jacobian_local_address + 3);

					for (int i_face_vertex = 0; i_face_vertex < 3; i_face_vertex++) {
						int i_vertex = face_anchor.vertices[i_face_vertex];
						if (i_vertex == -1) continue;
						int i_vertex_anchor = face_anchor.vertex_anchor_indices[i_face_vertex];
						// [3x3] warped vertex position Jacobian w.r.t. node rotation
						const Eigen::SkewSymmetricMatrix3<float> dv_drotation(
								Eigen::Map<const Eigen::Vector3f>(
										warped_vertex_position_jacobian_data +
										(i_vertex * anchor_count_per_vertex * 4) + // vertex index * stride
										(i_vertex_anchor * 4) // index of the anchor for this vertex * stride
								)
						);
						// used to compute warped vertex position Jacobian w.r.t. node translation, weight * I_3x3
						float stored_node_weight =
								warped_vertex_position_jacobian_data[
										(i_vertex * anchor_count_per_vertex * 4) +
										(i_vertex_anchor * 4) + 3];
						// [3x3] warped vertex normal Jacobian w.r.t. node rotation
						const Eigen::SkewSymmetricMatrix3<float> dn_drotation(
								Eigen::Map<const Eigen::Vector3f>(
										warped_vertex_normal_jacobian_data +
										(i_vertex * anchor_count_per_vertex * 3) +
										(i_vertex_anchor * 3)
								)
						);
						// [1x3]
						auto dr_dv = dr_dV.block<1, 3>(0, i_face_vertex * 3);
						// [1x3]
						auto dr_dn = dr_dN.block<1, 3>(0, i_face_vertex * 3);

						//__DEBUG
						// auto dr_dv_c = dr_dv.eval();
						// auto dr_dn_c = dr_dn.eval();
						// auto dv_drotation_c = dv_drotation.toDenseMatrix();
						// auto dn_drotation_c = dn_drotation.toDenseMatrix();

						// [1x3] = ([1x3] * [3x3]) + ([1x3] * [3x3])
						//__DEBUG (uncomment)
						// pixel_vertex_anchor_rotation_jacobian += (dr_dv * dv_drotation) + (dr_dn * dn_drotation);
						pixel_vertex_anchor_translation_jacobian +=
								dr_dv * (Eigen::Matrix3f::Identity() * stored_node_weight);
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

template<open3d::core::Device::DeviceType TDevice>
void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
		open3d::core::Tensor& node_jacobians,
		open3d::core::Tensor& node_jacobian_ranges,
		open3d::core::Tensor& node_pixel_indices,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_counts,
		const open3d::core::Tensor& pixel_jacobians
) {
	// === dimension, type, and device tensor checks ===
	int64_t node_count = node_pixel_counts.GetShape(0);
	o3c::Device device = node_pixel_counts.GetDevice();
	o3c::AssertTensorShape(node_pixel_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_counts, o3c::Int32);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);

	o3c::AssertTensorShape(pixel_jacobians, { utility::nullopt, utility::nullopt, 6 });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	// === get access to input arrays ===
	auto node_pixel_index_jagged_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_counts.GetDataPtr<int32_t>();
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();

	// === set up atomic counter ===
	NNRT_DECLARE_ATOMIC(uint32_t, total_jacobian_count);
	NNRT_INITIALIZE_ATOMIC(uint32_t, total_jacobian_count, 0L);

	// === set up output tensor to store ranges ===
	node_jacobian_ranges = o3c::Tensor({node_count, 2}, o3c::Int64, device);
	auto node_jacobian_range_data = node_jacobian_ranges.GetDataPtr<int64_t>();
	// === loop over all nodes and sort all entries by jacobian address
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				int* node_jacobian_list_start =
						node_pixel_index_jagged_data + (node_index * MAX_PIXELS_PER_NODE);
				int node_jacobian_list_length = node_pixel_count_data[node_index];
				// sort the anchor jacobian addresses
				HeapSort(node_jacobian_list_start, node_jacobian_list_length);

				node_jacobian_range_data[node_index * 2 + 1] = node_jacobian_list_length;
				NNRT_ATOMIC_ADD(total_jacobian_count, static_cast<uint32_t>(node_jacobian_list_length));
			}
	);

	node_jacobians = o3c::Tensor({NNRT_GET_ATOMIC_VALUE_HOST(total_jacobian_count), 6L}, o3c::Float32,
	                             device);

	NNRT_CLEAN_UP_ATOMIC(total_jacobian_count);
	auto node_jacobian_data = node_jacobians.GetDataPtr<float>();

	node_pixel_indices = o3c::Tensor({node_jacobians.GetShape(0)}, o3c::Int32, device);
	auto node_pixel_index_compact_data = node_pixel_indices.GetDataPtr<int32_t>();

	// === loop over all nodes again, this time aggregating their jacobians for each pixel they affect
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				// figure out where to start filling in the jacobians for the current node
				int node_jacobian_start_index = 0;
				for (int i_node = 0; i_node < node_index - 1; i_node++) {
					node_jacobian_start_index += node_jacobian_range_data[i_node * 2 + 1];
				}
				node_jacobian_range_data[node_index * 2] = node_jacobian_start_index;

				// source data to get anchor jacobians from
				int* node_pixel_index_jagged_start = node_pixel_index_jagged_data + (node_index * MAX_PIXELS_PER_NODE);
				int node_pixel_count = node_pixel_count_data[node_index];

				// loop over source data and fill in node jacobians & corresponding pixel indices
				int i_node_jacobian = 0;
				for (int i_node_pixel = 0; i_node_pixel < node_pixel_count; i_node_pixel++) {
					int pixel_jacobian_address = node_pixel_index_jagged_start[i_node_pixel];
					Eigen::Map<const Eigen::RowVector<float, 6>>
							pixel_jacobian(pixel_jacobian_data + pixel_jacobian_address);
					Eigen::Map<Eigen::RowVector<float, 6>>
							node_jacobian(node_jacobian_data + node_jacobian_start_index + i_node_jacobian * 6);
					node_pixel_index_compact_data[node_jacobian_start_index + i_node_jacobian]
							= pixel_jacobian_address / 6;
					node_jacobian = pixel_jacobian;
				}
			}
	);
}

// @formatter:off
NNRT_CONSTANT_WHEN_CUDACC const int column_0_lookup_table[21] = {
            0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
                  2, 2, 2, 2,
                     3, 3, 3,
                        4, 4,
                           5
};
NNRT_CONSTANT_WHEN_CUDACC const int column_1_lookup_table[21] = {
            0, 1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
                  2, 3, 4, 5,
                     3, 4, 5,
                        4, 5,
                           5
};
// @formatter:on



//TODO: can optimize: don't fill in lower triangle at all, use a batched triangular solver instead of Cholesky?
template<open3d::core::Device::DeviceType TDevice>
void ComputeHessianApproximationBlocks_UnorderedNodePixels(
		open3d::core::Tensor& hessian_approximation_blocks,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts
) {
	// === dimension, type, and device tensor checks ===
	int64_t node_count = node_pixel_jacobian_counts.GetShape(0);
	o3c::Device device = node_pixel_jacobian_counts.GetDevice();

	o3c::AssertTensorShape(node_pixel_jacobian_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_jacobian_counts, o3c::Int32);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);

	o3c::AssertTensorShape(pixel_jacobians, { utility::nullopt, utility::nullopt, 6 });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	// === get access to input arrays ===
	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_jacobian_counts.GetDataPtr<int32_t>();
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();

	/*
	 * Each node has 6 delta components (3 rotational and 3 translational).
	 * These would be represented in a dense jacobian as rows within the same 6 columns, with each row
	 * corresponding to a separate pixel.
	 * Block-diagonal optimization assumes that nodes have no influence on each
	 * other, so we can imagine separate J_node matrices for each node, of size [pixel count x 6]
	 * Each block can then be found via (J_node^T)*J_node, and will have size 6x6.
	 * Each entry in a single 6x6 block is a dot product of one of the six columns with another, hence:
	 *    1. The number of unique column combinations is (2 + 6 - 1)! / (2!*(6-1)!), or 21,
	 *       so there are only 21 unique entries per block
	 *    2. Due to commutativity of addition, it doesn't matter in what order the addends of each dot product are
	 *       retrieved from memory, as long as we always sample their factors from the same row
	 */
	// initialize output structures
	const int64_t unique_entry_count_per_block = 21;
	hessian_approximation_blocks = o3c::Tensor({node_count, 6, 6}, o3c::Float32, device);
	auto hessian_approximation_block_data = hessian_approximation_blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device, unique_entry_count_per_block * node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_index) {
				int i_node = static_cast<int>(workload_index / unique_entry_count_per_block);
				int i_unique_element_in_block = static_cast<int>(workload_index % unique_entry_count_per_block);
				int i_column_0 = column_0_lookup_table[i_unique_element_in_block];
				int i_column_1 = column_1_lookup_table[i_unique_element_in_block];
				int node_pixel_jacobian_list_length = node_pixel_count_data[i_node];
				float column_product = 0.0;
				for (int i_node_pixel_jacobian = 0; i_node_pixel_jacobian < node_pixel_jacobian_list_length; i_node_pixel_jacobian++) {
					int pixel_node_jacobian_address = node_pixel_jacobian_index_data[i_node * MAX_PIXELS_PER_NODE + i_node_pixel_jacobian];
					const float* jacobian = pixel_jacobian_data + pixel_node_jacobian_address;
					float addend = jacobian[i_column_0] * jacobian[i_column_1];
					column_product += addend;
				}
				// parenthesis for clarity
				hessian_approximation_block_data[(i_node * 36) + (i_column_0 * 6) + i_column_1] = column_product;
				hessian_approximation_block_data[(i_node * 36) + (i_column_1 * 6) + i_column_0] = column_product;
			}
	);
}

template<open3d::core::Device::DeviceType TDevice>
void ComputeNegativeGradient_UnorderedNodePixels(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex
) {
	// === dimension, type, and device tensor checks ===
	o3c::Device device = pixel_jacobians.GetDevice();
	int64_t pixel_count = pixel_jacobians.GetShape(0);
	int64_t max_jacobian_count = pixel_jacobians.GetShape(1);

	o3c::AssertTensorShape(pixel_jacobians, { pixel_count, max_jacobian_count, 6 });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	o3c::AssertTensorShape(residuals, { pixel_count });
	o3c::AssertTensorDtype(residuals, o3c::Float32);
	o3c::AssertTensorDevice(residuals, device);

	o3c::AssertTensorShape(residual_mask, { pixel_count });
	o3c::AssertTensorDtype(residual_mask, o3c::Bool);
	o3c::AssertTensorDevice(residual_mask, device);

	int64_t node_count = node_pixel_jacobian_counts.GetShape(0);
	o3c::AssertTensorShape(node_pixel_jacobian_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_jacobian_counts, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_counts, device);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);


	// === get access to input arrays ===
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();
	auto residual_data = residuals.GetDataPtr<float>();
	auto residual_mask_data = static_cast<const unsigned char*>(residual_mask.GetDataPtr());
	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_jacobian_counts.GetDataPtr<int32_t>();

	// === initialize output structures ===
	negative_gradient = o3c::Tensor::Zeros({node_count * 6}, o3c::Float32, device);
	auto negative_gradient_data = negative_gradient.GetDataPtr<float>();

	//TODO: this is extremely inefficient. There must be some parallel reduction we can apply here, but everything is difficult.
	// The current code does a variable number of essentially random (global) memory accesses for each node (as many as there are jacobians,
	// i.e. pixels controlled by each node.  These are ordered, potentially not at all cache-able. In addition, it makes as many random accesses
	// to the residual array in order to perform the actual multiplication. These results are summed together to produce entries of the gradient.
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				int node_pixel_jacobian_list_length = node_pixel_count_data[node_index];
				Eigen::Vector<float, 6> node_pixel_gradient;
				node_pixel_gradient << 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
				for (int i_node_pixel_jacobian = 0; i_node_pixel_jacobian < node_pixel_jacobian_list_length; i_node_pixel_jacobian++) {
					int pixel_node_jacobian_address = node_pixel_jacobian_index_data[node_index * MAX_PIXELS_PER_NODE + i_node_pixel_jacobian];
					int i_pixel = pixel_node_jacobian_address / (max_anchor_count_per_vertex * 3 * 6);
					//TODO: NOT sure mask filtering helps with anything here -- seems like it would only contribute to thread divergence
					if (!residual_mask_data[i_pixel]) continue;
					Eigen::Map<const Eigen::Vector<float, 6>> node_pixel_jacobian(pixel_jacobian_data + pixel_node_jacobian_address);
					float residual = residual_data[i_pixel];
					node_pixel_gradient -= node_pixel_jacobian * residual;
				}
				Eigen::Map<Eigen::Vector<float, 6>> negative_node_gradient(negative_gradient_data + node_index * 6);
				negative_node_gradient -= node_pixel_gradient;
			}
	);
}

} // namespace nnrt::alignment::kernel