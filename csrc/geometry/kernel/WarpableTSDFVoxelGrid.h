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
#include "geometry/kernel/DeviceSelection.h"

#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/Hashmap.h>
#include <open3d/core/hashmap/HashmapBuffer.h>


namespace nnrt::geometry::kernel::tsdf {
// region ================ DUAL QUATERNION VERSIONS ============================

template<AnchorComputationMethod TAnchorComputationMethod, bool TUseNodeDistanceThreshold, open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                       open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                       const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                       const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                       const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                       const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                       int minimum_valid_anchor_count, float depth_scale, float depth_max);

template<AnchorComputationMethod TAnchorComputationMethod, bool TUseNodeDistanceThreshold>
void IntegrateWarpedDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                       open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                       const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                       const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                       const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                       const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                       int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	InferDeviceFromTensorAndExecute(
			block_keys,
			[&] {
				IntegrateWarpedDQ<TAnchorComputationMethod, TUseNodeDistanceThreshold, open3d::core::Device::DeviceType::CPU>(
						block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
						depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
						node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
				);
			},
			[&] {
				IntegrateWarpedDQ<TAnchorComputationMethod, TUseNodeDistanceThreshold, open3d::core::Device::DeviceType::CUDA>(
						block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
						depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
						node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
				);
			}
	);
}


// endregion ===================================================================
// region ================== MATRIX VERSIONS ===================================

template<AnchorComputationMethod TAnchorComputationMethod, bool TUseNodeDistanceThreshold, open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                        open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                        const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                        const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                        int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max);

template<AnchorComputationMethod TAnchorComputationMethod, bool TUseNodeDistanceThreshold>
void IntegrateWarpedMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                        open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                        const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                        const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
                        int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	InferDeviceFromTensorAndExecute(
			block_keys,
			[&] {
				IntegrateWarpedMat<TAnchorComputationMethod, TUseNodeDistanceThreshold, open3d::core::Device::DeviceType::CPU>(
						block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
						depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
						node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
				);
			},
			[&] {
				IntegrateWarpedMat<TAnchorComputationMethod, TUseNodeDistanceThreshold, open3d::core::Device::DeviceType::CUDA>(
						block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
						depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
						node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
				);
			}
	);
}

// endregion ==================================================================

// TODO: implement (then, maybe, provide dual-quaternion version ?)
// void DetermineWhichBlocksToActivateWithWarp(open3d::core::Tensor& blocks_to_activate_mask, const open3d::core::Tensor& candidate_block_coordinates,
//                                             const open3d::core::Tensor& depth_downsampled, const open3d::core::Tensor& intrinsics_downsampled,
//                                             const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
//                                             const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
//                                             int64_t block_resolution, float voxel_size, float sdf_truncation_distance);
//
// template<open3d::core::Device::DeviceType TDeviceType>
// void DetermineWhichBlocksToActivateWithWarp(open3d::core::Tensor& blocks_to_activate_mask, const open3d::core::Tensor& candidate_block_coordinates,
//                                             const open3d::core::Tensor& depth_downsampled, const open3d::core::Tensor& intrinsics_downsampled,
//                                             const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
//                                             const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, float node_coverage,
//                                             int64_t block_resolution, float voxel_size, float sdf_truncation_distance);


} // namespace nnrt::geometry::kernel::tsdf



