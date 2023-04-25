//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/25/22.
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
#include <open3d/core/hashmap/HashMap.h>

namespace nnrt::geometry::functional {


/**
 * \brief average-downsample points within each grid cell of a uniform cubic grid
 * \param downsampled_points resulting points
 * \param original_points original points
 * \param grid_cell_size side length of each (cubic) grid cell
 * \param hash_backend hash backend to use for the operation
 */
open3d::core::Tensor GridAverageDownsample3dPoints(const open3d::core::Tensor& original_points, float grid_cell_size,
                                                   const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default);

/**
 * \brief average-downsample points such that resulting points are at least at min_distance apart.
 * \details does not necessarily produce a maximal downsampling set, but efficient since it uses two grid-averaging
 * downsampling iterations (with a 1/2 grid-cell offset in the second iteration) to produce the result.
 * \param downsampled_points resulting downsampled points
 * \param original_points original points
 * \param min_distance minimal distance between resulting points
 * \param hash_backend backend to use for hashing
 */
open3d::core::Tensor FastRadiusAverageDownsample3dPoints(const open3d::core::Tensor& original_points, float radius,
                                                         const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default);



open3d::core::Tensor
RadiusMedianSubsample3dPoints(const open3d::core::Tensor& original_points, float radius, const open3d::core::HashBackendType& hash_backend_type);

} // nnrt::geometry::functional