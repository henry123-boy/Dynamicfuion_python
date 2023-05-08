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

#include <open3d/core/Dispatch.h>
#include <open3d/core/Tensor.h>
#include <open3d/core/ParallelFor.h>
#include "core/heap/CPU/DeviceHeapCPU.h"
#include "geometry/kernel/NonRigidSurfaceVoxelBlockGridImpl.h"


using namespace open3d;
namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::voxel_grid {

template
void IntegrateNonRigid<open3d::core::Device::DeviceType::CPU>(
		const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, t::geometry::TensorMap& block_value_map,
		open3d::core::Tensor& cos_voxel_ray_to_normal, index_t block_resolution, float voxel_size, float sdf_truncation_distance,
		const open3d::core::Tensor& depth, const open3d::core::Tensor& color, const open3d::core::Tensor& depth_normals,
		const open3d::core::Tensor& depth_intrinsics, const open3d::core::Tensor& color_intrinsics, const open3d::core::Tensor& extrinsics,
		const WarpField& warp_field, float depth_scale, float depth_max
);

template
void GetBoundingBoxesOfWarpedBlocks<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_keys, const WarpField& warp_field, float voxel_size,
		index_t block_resolution, const open3d::core::Tensor& extrinsics
);

template
void GetAxisAlignedBoxesInterceptingSurfaceMask<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& mask, const open3d::core::Tensor& boxes, const open3d::core::Tensor& intrinsics,
		const open3d::core::Tensor& depth, float depth_scale, float depth_max, int32_t stride, float truncation_distance
);

template
void ExtractVoxelValuesAndCoordinates<open3d::core::Device::DeviceType::CPU>(
		o3c::Tensor& voxel_values_and_coordinates, const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys,
		const open3d::t::geometry::TensorMap& block_value_map, int64_t block_resolution, float voxel_size
);

template
void ExtractVoxelValuesAt<open3d::core::Device::DeviceType::CPU>(
		o3c::Tensor& voxel_values, const o3c::Tensor& query_coordinates, const open3d::core::Tensor& query_block_indices,
		const open3d::core::Tensor& block_keys, const open3d::t::geometry::TensorMap& block_value_map, int64_t block_resolution, float voxel_size
);

} // namespace nnrt::geometry::kernel::tsdf