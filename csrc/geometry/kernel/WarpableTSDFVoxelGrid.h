//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
//  Copyright (c) 2021 Gregory Kramida
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

#include "geometry/AnchorComputationMethod.h"
#include "core/DeviceSelection.h"

#include <open3d/core/Tensor.h>
#include "geometry/GraphWarpField.h"


namespace nnrt::geometry::kernel::tsdf {

void IntegrateWarped(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                     open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                     const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                     const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
                     float depth_scale, float depth_max);

template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarped(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                     open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                     const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                     const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
                     float depth_scale, float depth_max);

void GetBoundingBoxesOfWarpedBlocks(open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_keys, const GraphWarpField& warp_field,
                               float voxel_size, int64_t block_resolution, const open3d::core::Tensor& extrinsics);

template<open3d::core::Device::DeviceType TDeviceType>
void GetBoundingBoxesOfWarpedBlocks(open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_keys, const GraphWarpField& warp_field,
                               float voxel_size, int64_t block_resolution, const open3d::core::Tensor& extrinsics);

void GetAxisAlignedBoxesInterceptingSurfaceMask(open3d::core::Tensor& mask, const open3d::core::Tensor& boxes, const open3d::core::Tensor& intrinsics,
                                                const open3d::core::Tensor& depth, float depth_scale, float depth_max, int32_t stride,
                                                float truncation_distance);

template<open3d::core::Device::DeviceType TDeviceType>
void GetAxisAlignedBoxesInterceptingSurfaceMask(open3d::core::Tensor& mask, const open3d::core::Tensor& boxes, const open3d::core::Tensor& intrinsics,
                                                const open3d::core::Tensor& depth, float depth_scale, float depth_max, int32_t stride,
                                                float truncation_distance);


} // namespace nnrt::geometry::kernel::tsdf



