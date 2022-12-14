//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/18/22.
//  Copyright (c) 2022 Gregory Kramida
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
// 3rd party
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

//local
#include "rendering/kernel/ExtractClippedFaceVertices.h"
#include "core/PlatformIndependentAtomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;


namespace nnrt::rendering::kernel {

template<open3d::core::Device::DeviceType TDeviceType, bool TExtractNormals, typename TRunBeforeReturnInvalid, typename TGetOutputIndex>
void ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic(
		open3d::core::Tensor& vertex_positions_normalized_camera,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_vertex_normals_camera,
		const open3d::core::Tensor& vertex_positions_camera,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> vertex_normals_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& ndc_intrinsic_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance,
		TRunBeforeReturnInvalid&& run_before_return_invalid,
		TGetOutputIndex&& get_output_face_index
) {
	o3c::Device device = vertex_positions_camera.GetDevice();

	auto face_count = triangle_vertex_indices.GetLength();
	// input indexers
	o3tgk::NDArrayIndexer face_vertex_index_indexer(triangle_vertex_indices, 1);
	o3tgk::NDArrayIndexer vertex_position_indexer(vertex_positions_camera, 1);

	// output & output indexers
	vertex_positions_normalized_camera = open3d::core::Tensor({face_count, 3, 3}, o3c::Float32, device);
	o3tgk::NDArrayIndexer normalized_face_vertex_indexer(vertex_positions_normalized_camera, 1);
	o3tgk::TransformIndexer perspective_transform(ndc_intrinsic_matrix, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0f);

	float* face_vertex_normal_ptr = nullptr;
	const float* vertex_normal_ptr = nullptr;
	if (TExtractNormals) {
		face_vertex_normals_camera.value().get() = open3d::core::Tensor({face_count, 3, 3}, o3c::Float32, device);
		face_vertex_normal_ptr = face_vertex_normals_camera.value().get().GetDataPtr<float>();
		vertex_normal_ptr = vertex_normals_camera.value().get().GetDataPtr<float>();
	}


	o3c::ParallelFor(
			device, face_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t face_idx) {
				Eigen::Map<Eigen::Matrix<int64_t, 1, 3>> vertex_indices(face_vertex_index_indexer.GetDataPtr<int64_t>(face_idx));

				Eigen::Map<Eigen::Vector3f> vertex0(vertex_position_indexer.GetDataPtr<float>(vertex_indices(0)));
				Eigen::Map<Eigen::Vector3f> vertex1(vertex_position_indexer.GetDataPtr<float>(vertex_indices(1)));
				Eigen::Map<Eigen::Vector3f> vertex2(vertex_position_indexer.GetDataPtr<float>(vertex_indices(2)));
				Eigen::Map<Eigen::Vector3f> vertices[] = {vertex0, vertex1, vertex2};

				// clip if all vertices are too near or too far
				bool have_vertex_within_clipping_range = false;
				for (auto& vertex: vertices) {
					have_vertex_within_clipping_range |= vertex.z() >= near_clipping_distance;
					have_vertex_within_clipping_range |= vertex.z() <= far_clipping_distance;
				}
				if (!have_vertex_within_clipping_range) {
					run_before_return_invalid(face_idx);
					return;
				}

				Eigen::Vector2f normalized_face_vertices_xy[3];
				bool has_inliner_vertex = false;
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					perspective_transform.Project(vertices[i_vertex].x(), vertices[i_vertex].y(), vertices[i_vertex].z(),
					                              &normalized_face_vertices_xy[i_vertex].x(), &normalized_face_vertices_xy[i_vertex].y());
					has_inliner_vertex |= normalized_camera_space_xy_range.Contains(normalized_face_vertices_xy[i_vertex]);
				}
				if (!has_inliner_vertex) {
					run_before_return_invalid(face_idx);
					return;
				}

				auto output_face_index = get_output_face_index(face_idx);
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex0(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index));
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex1(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 3);
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex2(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 6);
				Eigen::Map<Eigen::Vector3f> normalized_face_vertices[] = {normalized_face_vertex0, normalized_face_vertex1, normalized_face_vertex2};
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					normalized_face_vertices[i_vertex].x() = normalized_face_vertices_xy[i_vertex].x();
					// important: flip the y-coordinate to reflect pixel space
					normalized_face_vertices[i_vertex].y() = normalized_face_vertices_xy[i_vertex].y();
					normalized_face_vertices[i_vertex].z() = vertices[i_vertex].z();
				}
				if (TExtractNormals) {
					for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
						memcpy(face_vertex_normal_ptr + output_face_index * 9 + i_vertex * 3,
						       vertex_normal_ptr + (vertex_indices(i_vertex) * 3),
						       sizeof(float) * 3);
					}
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void MeshVerticesClippedToNdc(open3d::core::Tensor& vertex_positions_normalized_camera,
                              const open3d::core::Tensor& vertex_positions_camera,
                              const open3d::core::Tensor& triangle_vertex_indices,
                              const open3d::core::Tensor& normalized_camera_space_matrix,
                              kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
                              float near_clipping_distance,
                              float far_clipping_distance) {
	NNRT_DECLARE_ATOMIC_INT(unclipped_face_count);
	NNRT_INITIALIZE_ATOMIC(int, unclipped_face_count, 0);
	auto run_before_return_invalid = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) { return; };
	auto get_output_face_index = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		return static_cast<int64_t>(NNRT_ATOMIC_ADD(unclipped_face_count, 1));
	};
	ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic<TDeviceType, false>(
			vertex_positions_normalized_camera, utility::nullopt, vertex_positions_camera, utility::nullopt,
			triangle_vertex_indices, normalized_camera_space_matrix,
			normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance,
			run_before_return_invalid, get_output_face_index
	);
	int unclipped_face_count_host = NNRT_GET_ATOMIC_VALUE_HOST(unclipped_face_count);
	vertex_positions_normalized_camera = vertex_positions_normalized_camera.Slice(0, 0, unclipped_face_count_host);
}

template<open3d::core::Device::DeviceType TDeviceType>
void MeshDataAndClippingMaskToNdc(open3d::core::Tensor& vertex_positions_normalized_camera,
                                  open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_vertex_normals_camera,
                                  open3d::core::Tensor& clipped_face_mask, const open3d::core::Tensor& vertex_positions_camera,
                                  open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> vertex_normals_camera,
                                  const open3d::core::Tensor& triangle_vertex_indices,
                                  const open3d::core::Tensor& normalized_camera_space_matrix,
                                  kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range, float near_clipping_distance,
                                  float far_clipping_distance) {
	auto face_count = triangle_vertex_indices.GetLength();
	o3c::Device device = vertex_positions_camera.GetDevice();

	if (face_vertex_normals_camera.has_value() != vertex_normals_camera.has_value()) {
		utility::LogError("either both or none of face_vertex_normals_camera[out] and normals_camera[in] need to have Tensor values passed in. "
		              "face_vertex_normals_camera: {}; normals_camera: {}}", (face_vertex_normals_camera.has_value() ? " has value" : " is null"),
		                  (vertex_normals_camera.has_value() ? " has value" : " is null"));
	}
	bool extract_normals = vertex_normals_camera.has_value();

	clipped_face_mask = open3d::core::Tensor::Ones({face_count}, o3c::Bool, device);
	bool* clipped_face_mask_ptr = clipped_face_mask.template GetDataPtr<bool>();
	auto run_before_return_invalid = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		clipped_face_mask_ptr[workload_idx] = false;
	};
	auto get_output_face_index = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		return workload_idx;
	};
	if (extract_normals) {
		ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic<TDeviceType, true>(
				vertex_positions_normalized_camera, face_vertex_normals_camera, vertex_positions_camera, vertex_normals_camera,
				triangle_vertex_indices, normalized_camera_space_matrix, normalized_camera_space_xy_range, near_clipping_distance,
				far_clipping_distance, run_before_return_invalid, get_output_face_index
		);
	} else {
		ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic<TDeviceType, false>(
				vertex_positions_normalized_camera, face_vertex_normals_camera, vertex_positions_camera, vertex_normals_camera,
				triangle_vertex_indices, normalized_camera_space_matrix, normalized_camera_space_xy_range, near_clipping_distance,
				far_clipping_distance, run_before_return_invalid, get_output_face_index
		);
	}


}

} // namespace nnrt::rendering::kernel;