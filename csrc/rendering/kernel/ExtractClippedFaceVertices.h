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
#include <open3d/core/Tensor.h>
#include "AxisAlignedBoundingBox.h"

namespace nnrt::rendering::kernel {
using t_image_index = int32_t;

void ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_normalized_camera,
		open3d::core::Tensor& clipped_face_mask,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_intrinsic_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
);

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_normalized_camera,
		open3d::core::Tensor& clipped_face_mask,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_intrinsic_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
);

void ExtractClippedFaceVerticesInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_camera_space_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
);

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractClippedFaceVerticesInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_camera_space_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
);
} // namespace nnrt::rendering::kernel