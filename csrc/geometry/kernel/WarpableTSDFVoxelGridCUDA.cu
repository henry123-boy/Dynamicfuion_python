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
#include <open3d/core/hashmap/CUDA/StdGPUHashmap.h>
#include <open3d/core/hashmap/DeviceHashmap.h>
#include <open3d/core/hashmap/Dispatch.h>
#include <open3d/core/kernel/CUDALauncher.cuh>
#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/WarpableTSDFVoxelGridImpl.h"

using namespace open3d;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {


template
void IntegrateWarpedDQ<core::Device::DeviceType::CUDA>(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                                                       core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size,
                                                       float sdf_truncation_distance,
                                                       const core::Tensor& depth_tensor, const core::Tensor& color_tensor,
                                                       const core::Tensor& depth_normals,
                                                       const core::Tensor& intrinsics, const core::Tensor& extrinsics,
                                                       const core::Tensor& warp_graph_nodes,
                                                       const core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                       int anchor_count,
                                                       const int minimum_valid_anchor_count, float depth_scale, float depth_max);

template
void IntegrateWarpedMat<core::Device::DeviceType::CUDA>(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                                                        core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size,
                                                        float sdf_truncation_distance,
                                                        const core::Tensor& depth_tensor, const core::Tensor& color_tensor,
                                                        const core::Tensor& depth_normals,
                                                        const core::Tensor& intrinsics, const core::Tensor& extrinsics,
                                                        const core::Tensor& graph_nodes,
                                                        const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                                        const float node_coverage, const int anchor_count,
                                                        const int minimum_valid_anchor_count, const float depth_scale, const float depth_max);

template
void TouchWarpedMat<core::Device::DeviceType::CUDA>(std::shared_ptr<open3d::core::Hashmap>& hashmap,
                                                   const open3d::core::Tensor& points,
                                                   open3d::core::Tensor& voxel_block_coords,
                                                   int64_t voxel_grid_resolution,
                                                   const open3d::core::Tensor& extrinsics,
                                                   const open3d::core::Tensor& warp_graph_nodes,
                                                   const open3d::core::Tensor& node_rotations,
                                                   const open3d::core::Tensor& node_translations,
                                                   float node_coverage,
                                                   float voxel_size,
                                                   float sdf_trunc);

} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt