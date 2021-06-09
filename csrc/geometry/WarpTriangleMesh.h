//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/21.
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

#include <open3d/t/geometry/TriangleMesh.h>

namespace nnrt {
namespace geometry {

open3d::t::geometry::TriangleMesh WarpTriangleMeshMat(const open3d::t::geometry::TriangleMesh& input_mesh,
                                                      const open3d::core::Tensor& nodes,
                                                      const open3d::core::Tensor& node_rotations,
                                                      const open3d::core::Tensor& node_translations,
                                                      int anchor_count,
                                                      float node_coverage);

} // namespace geometry
} // namespace nnrt