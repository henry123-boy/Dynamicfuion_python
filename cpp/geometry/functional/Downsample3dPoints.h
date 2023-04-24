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


/// Downsample the current set of points by averaging them within a grid.
open3d::core::Tensor GridDownsample3dPoints(const open3d::core::Tensor& original_points, float grid_cell_size,
                                            const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default);

/// Ensures no two points in the result point cloud are within the specified radius. Note: does not produce the maximal set of such points.
open3d::core::Tensor RadiusDownsample3dPoints(const open3d::core::Tensor& original_points, float radius,
                                              const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default);

} // nnrt::geometry::functional