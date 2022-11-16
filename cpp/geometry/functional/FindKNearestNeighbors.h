//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/2/22.
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

namespace nnrt::geometry{

void FindKNearest3DPoints(open3d::core::Tensor& neighbor_indices_kdtree, open3d::core::Tensor& neighbor_distances_kdtree,
						  const open3d::core::Tensor& reference_points, const open3d::core::Tensor& query_points, int k);
open3d::core::Tensor FindKNearest3DPoints(const open3d::core::Tensor& reference_points, const open3d::core::Tensor& query_points, int k);

void FindKNearest3DNeighbors(open3d::core::Tensor& neighbor_indices_kdtree, open3d::core::Tensor& neighbor_distances_kdtree,
                                             const open3d::core::Tensor& points, int k);

open3d::core::Tensor FindKNearest3DNeighbors(const open3d::core::Tensor& points, int k);

} // nnrt::geometry