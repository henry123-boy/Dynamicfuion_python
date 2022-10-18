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
#include "rendering/kernel/ExtractClippedFaceVertices.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace nnrt::rendering::kernel {


void ExtractClippedFaceVerticesInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_camera_space_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
) {
	core::ExecuteOnDevice(
			vertex_positions_camera.GetDevice(),
			[&] {
				ExtractClippedFaceVerticesInNormalizedCameraSpace<o3c::Device::DeviceType::CPU>(
						vertex_positions_clipped_normalized_camera, vertex_positions_camera, triangle_vertex_indices, normalized_camera_space_matrix,
						normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ExtractClippedFaceVerticesInNormalizedCameraSpace<o3c::Device::DeviceType::CUDA>(
								vertex_positions_clipped_normalized_camera, vertex_positions_camera, triangle_vertex_indices,
								normalized_camera_space_matrix, normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
						);
				);
			}
	);
}


void ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_normalized_camera,
		open3d::core::Tensor& clipped_face_mask,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_camera_space_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance, float far_clipping_distance
) {
	core::ExecuteOnDevice(
			vertex_positions_camera.GetDevice(),
			[&] {
				ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace<o3c::Device::DeviceType::CPU>(
						vertex_positions_normalized_camera, clipped_face_mask, vertex_positions_camera, triangle_vertex_indices,
						normalized_camera_space_matrix,
						normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace<o3c::Device::DeviceType::CUDA>(
								vertex_positions_normalized_camera, clipped_face_mask, vertex_positions_camera, triangle_vertex_indices,
								normalized_camera_space_matrix, normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
						);
				);
			}
	);
}
} // namespace nnrt::rendering::kernel