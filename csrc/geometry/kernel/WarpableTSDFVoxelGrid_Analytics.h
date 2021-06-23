//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/23/21.
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
