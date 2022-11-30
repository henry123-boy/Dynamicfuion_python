//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>

// local includes
#include "geometry/GraphWarpField.h"

namespace nnrt::rendering::functional {

struct PartialDifferentiationState {
	open3d::core::Tensor rendered_normals;
	open3d::core::Tensor reference_to_rendered_vectors;
};

std::tuple<open3d::core::Tensor, PartialDifferentiationState>
ComputeReferencePointToRenderedPlaneDistances(const nnrt::geometry::GraphWarpField& warp_field,
                                              const open3d::t::geometry::TriangleMesh& canonical_mesh,
                                              const open3d::t::geometry::Image& reference_color_image,
                                              const open3d::t::geometry::PointCloud& reference_point_cloud,
                                              const open3d::core::Tensor& intrinsics,
                                              const open3d::core::Tensor& extrinsics,
                                              const open3d::core::Tensor& anchors,
                                              const open3d::core::Tensor& weights);
} // namespace nnrt::rendering::functional