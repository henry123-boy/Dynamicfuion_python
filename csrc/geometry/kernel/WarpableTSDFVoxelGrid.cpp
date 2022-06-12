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
#include <open3d/core/Tensor.h>
#include <open3d/utility/Console.h>
// #include <open3d/core/hashmap/Hashmap.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"

using namespace open3d;

namespace nnrt::geometry::kernel::tsdf {


void IntegrateWarped(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                     open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                     const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                     const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
                     float depth_scale, float depth_max) {
	core::InferDeviceFromEntityAndExecute(
			block_keys,
			[&] {
				IntegrateWarped<open3d::core::Device::DeviceType::CPU>(
						block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
						depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_field, depth_scale, depth_max
				);
			},
			[&] {
				NNRT_IF_CUDA(
						IntegrateWarped<open3d::core::Device::DeviceType::CUDA>(
								block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size,
								sdf_truncation_distance, depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_field, depth_scale,
								depth_max
						);
				);
			}
	);
}

void GetBoundingBoxesOfWarpedBlocks(open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_addresses,
                                    const GraphWarpField& warp_field, float voxel_size, int64_t block_resolution,
                                    const open3d::core::Tensor& extrinsics) {
	core::ExecuteOnDevice(
			block_addresses.GetDevice(),
			[&] {
				GetBoundingBoxesOfWarpedBlocks<open3d::core::Device::DeviceType::CPU>(bounding_boxes, block_addresses, warp_field, voxel_size,
				                                                                      block_resolution, extrinsics);
			},
			[&] {
				NNRT_IF_CUDA(
						GetBoundingBoxesOfWarpedBlocks<open3d::core::Device::DeviceType::CUDA>(bounding_boxes, block_addresses, warp_field,
						                                                                       voxel_size, block_resolution, extrinsics);
				);
			}
	);
}

void GetAxisAlignedBoxesInterceptingSurfaceMask(open3d::core::Tensor& mask, const open3d::core::Tensor& boxes, const open3d::core::Tensor& intrinsics,
                                                const open3d::core::Tensor& depth, float depth_scale, float depth_max, int32_t stride,
                                                float truncation_distance) {
	core::ExecuteOnDevice(
			boxes.GetDevice(),
			[&] {
				GetAxisAlignedBoxesInterceptingSurfaceMask<open3d::core::Device::DeviceType::CPU>(
						mask, boxes, intrinsics, depth, depth_scale, depth_max, stride, truncation_distance
				);
			},
			[&] {
				NNRT_IF_CUDA(
						GetAxisAlignedBoxesInterceptingSurfaceMask<open3d::core::Device::DeviceType::CUDA>(
								mask, boxes, intrinsics, depth, depth_scale, depth_max, stride, truncation_distance
						);
				);
			}

	);
}


} // namespace nnrt::geometry::kernel::tsdf