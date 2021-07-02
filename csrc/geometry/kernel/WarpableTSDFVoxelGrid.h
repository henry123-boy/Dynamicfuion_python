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

#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/Hashmap.h>
#include <open3d/core/hashmap/HashmapBuffer.h>

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

void IntegrateWarpedEuclideanDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                                open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                                const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                                const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                                int minimum_valid_anchor_count, float depth_scale, float depth_max);

template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedEuclideanDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                                open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                                const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                                const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                                int minimum_valid_anchor_count, float depth_scale, float depth_max);

void IntegrateWarpedEuclideanMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                                 open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                 const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                                 const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                                 const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                                 int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max);

template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedEuclideanMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                                 open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                 const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                                 const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                                 const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                                 int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max);


//TODO: provide dual-quaternion version (?)
void DetermineWhichBlocksToActivateWithWarp(open3d::core::Tensor& blocks_to_activate_mask, const open3d::core::Tensor& candidate_block_coordinates,
                                            const open3d::core::Tensor& depth_downsampled, const open3d::core::Tensor& intrinsics_downsampled,
                                            const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                                            const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                                            int64_t block_resolution, float voxel_size, float sdf_truncation_distance);

template<open3d::core::Device::DeviceType TDeviceType>
void DetermineWhichBlocksToActivateWithWarp(open3d::core::Tensor& blocks_to_activate_mask, const open3d::core::Tensor& candidate_block_coordinates,
                                            const open3d::core::Tensor& depth_downsampled, const open3d::core::Tensor& intrinsics_downsampled,
                                            const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                                            const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                                            int64_t block_resolution, float voxel_size, float sdf_truncation_distance);


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt



