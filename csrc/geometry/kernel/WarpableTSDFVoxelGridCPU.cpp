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
// COPY the missing file "open3d/core/hashmap/CPU/CPUHashmapBufferAccessor.hpp" from source to Open3D install folder manually.
// I don't know why it's not properly installed.
// #include <open3d/core/hashmap/CPU/TBBHashmap.h>
// #include <open3d/core/hashmap/Dispatch.h>
#include <open3d/core/kernel/CPULauncher.h>

#include "geometry/kernel/WarpableTSDFVoxelGridImpl.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid_AnalyticsImpl.h"


using namespace open3d;
namespace o3c = open3d::core;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

template
void IntegrateWarpedDQ<o3c::Device::DeviceType::CPU>(const o3c::Tensor& block_indices, const o3c::Tensor& block_keys, o3c::Tensor& block_values,
                                                      o3c::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size,
                                                      float sdf_truncation_distance,
                                                      const o3c::Tensor& depth_tensor, const o3c::Tensor& color_tensor,
                                                      const o3c::Tensor& depth_normals,
                                                      const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics,
                                                      const o3c::Tensor& warp_graph_nodes,
                                                      const o3c::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                                                      const int minimum_valid_anchor_count, float depth_scale, float depth_max);

template
void IntegrateWarpedMat<o3c::Device::DeviceType::CPU>(const o3c::Tensor& block_indices, const o3c::Tensor& block_keys, o3c::Tensor& block_values,
                                                       o3c::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size,
                                                       float sdf_truncation_distance,
                                                       const o3c::Tensor& depth_tensor, const o3c::Tensor& color_tensor,
                                                       const o3c::Tensor& depth_normals,
                                                       const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics,
                                                       const o3c::Tensor& graph_nodes,
                                                       const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                                                       const float node_coverage, const int anchor_count,
                                                       const int minimum_valid_anchor_count, const float depth_scale, const float depth_max);

template
void DetermineWhichBlocksToActivateWithWarp<o3c::Device::DeviceType::CPU>(
		o3c::Tensor& blocks_to_activate_mask, const o3c::Tensor& candidate_block_coordinates,
		const o3c::Tensor& depth_downsampled, const o3c::Tensor& intrinsics_downsampled,
		const o3c::Tensor& extrinsics, const o3c::Tensor& graph_nodes,
		const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations, float node_coverage,
		int64_t block_resolution, float voxel_size, float sdf_truncation_distance);

} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt