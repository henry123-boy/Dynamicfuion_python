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

// region ============================ device-routing (top-level) ==========================
void ExtractVoxelCenters(const open3d::core::Tensor& block_indices,
                         const open3d::core::Tensor& block_keys,
                         const open3d::core::Tensor& block_values,
                         open3d::core::Tensor& voxel_centers,
                         int64_t block_resolution,
                         float voxel_size);

void ExtractTSDFValuesAndWeights(const open3d::core::Tensor& block_indices,
                                 const open3d::core::Tensor& block_values,
                                 open3d::core::Tensor& voxel_values,
                                 int64_t block_resolution);

void ExtractValuesInExtent(int64_t min_voxel_x, int64_t min_voxel_y, int64_t min_voxel_z,
                           int64_t max_voxel_x, int64_t max_voxel_y, int64_t max_voxel_z,
                           const open3d::core::Tensor& block_indices,
                           const open3d::core::Tensor& block_keys,
                           const open3d::core::Tensor& block_values,
                           open3d::core::Tensor& voxel_values,
                           int64_t block_resolution);

void IntegrateWarpedDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                       open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                       const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                       const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                       const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                       const int minimum_valid_anchor_count, float depth_scale, float depth_max);

void IntegrateWarpedMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                        open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                        const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, const float node_coverage,
                        const int anchor_count, const int minimum_valid_anchor_count, const float depth_scale, const float depth_max);

// endregion
// region =============================== CPU ======================================
void ExtractVoxelCentersCPU(const open3d::core::Tensor& block_indices,
                            const open3d::core::Tensor& block_keys,
                            const open3d::core::Tensor& block_values,
                            open3d::core::Tensor& voxel_centers,
                            int64_t block_resolution, float voxel_size);

void ExtractTSDFValuesAndWeightsCPU(const open3d::core::Tensor& block_indices,
                                    const open3d::core::Tensor& block_values,
                                    open3d::core::Tensor& voxel_values,
                                    int64_t block_resolution);

void ExtractValuesInExtentCPU(int64_t min_x, int64_t min_y, int64_t min_z,
                              int64_t max_x, int64_t max_y, int64_t max_z,
                              const open3d::core::Tensor& block_indices,
                              const open3d::core::Tensor& block_keys,
                              const open3d::core::Tensor& block_values,
                              open3d::core::Tensor& voxel_values,
                              int64_t block_resolution);

template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedDQ(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                       open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                       const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                       const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                       const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                       const int minimum_valid_anchor_count, float depth_scale, float depth_max);
template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarpedMat(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                        open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& graph_nodes,
                        const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations, const float node_coverage,
                        const int anchor_count, const int minimum_valid_anchor_count, const float depth_scale, const float depth_max);

void TouchWarpedMat(std::shared_ptr<open3d::core::Hashmap>& hashmap,
                    const open3d::core::Tensor& points,
                    open3d::core::Tensor& voxel_block_coords,
                    const open3d::core::Tensor& extrinsics,
                    const open3d::core::Tensor& warp_graph_nodes,
                    const open3d::core::Tensor& node_rotations,
                    const open3d::core::Tensor& node_translations,
                    int64_t voxel_grid_resolution,
                    float node_coverage,
                    float voxel_size,
                    float sdf_trunc);

template<open3d::core::Device::DeviceType TDeviceType>
void TouchWarpedMat(std::shared_ptr<open3d::core::Hashmap>& hashmap,
                    const open3d::core::Tensor& points,
                    open3d::core::Tensor& voxel_block_coords,

                    const open3d::core::Tensor& extrinsics,
                    const open3d::core::Tensor& warp_graph_nodes,
                    const open3d::core::Tensor& node_rotations,
                    const open3d::core::Tensor& node_translations,
                    float node_coverage,
                    int64_t voxel_grid_resolution,
                    float voxel_size,
                    float sdf_trunc);

// endregion
// region =============================== CUDA ======================================
#ifdef BUILD_CUDA_MODULE
void ExtractVoxelCentersCUDA(const open3d::core::Tensor& block_indices,
                             const open3d::core::Tensor& block_keys,
                             const open3d::core::Tensor& block_values,
                             open3d::core::Tensor& voxel_centers,
                             int64_t block_resolution,
                             float voxel_size);

void ExtractTSDFValuesAndWeightsCUDA(const open3d::core::Tensor& block_indices,
                                     const open3d::core::Tensor& block_values,
                                     open3d::core::Tensor& voxel_values,
                                     int64_t block_resolution);

void ExtractValuesInExtentCUDA(int64_t min_x, int64_t min_y, int64_t min_z,
                               int64_t max_x, int64_t max_y, int64_t max_z,
                               const open3d::core::Tensor& block_indices,
                               const open3d::core::Tensor& block_keys,
                               const open3d::core::Tensor& block_values,
                               open3d::core::Tensor& voxel_values,
                               int64_t block_resolution);
#endif
// endregion



} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt



