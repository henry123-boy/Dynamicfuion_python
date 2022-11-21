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
// stdlib includes

// third-party includes

// local includes
#include "rendering/functional/kernel/PartialDerivatives.h"
#include "rendering/functional/PartialDerivatives.h"


namespace utility = open3d::utility;
namespace o3c = open3d::core;

// local includes
namespace nnrt::rendering::functional {
std::tuple <open3d::core::Tensor, open3d::core::Tensor>
WarpedVertexAndNormalJacobians(const open3d::t::geometry::TriangleMesh& warped_mesh, const geometry::GraphWarpField& warp_field,
                               const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights) {
	if (!warped_mesh.HasVertexNormals() || !warped_mesh.HasVertexPositions()) {
		utility::LogError("warped_mesh needs to have both vertex positions and vertex normals defined. In argument, vertex positions are {} defined, "
		                  "and vertex normals are {} defined.", (warped_mesh.HasVertexPositions() ? "" : "not"),
		                  +(warped_mesh.HasVertexNormals() ? "" : "not"));
	}

	o3c::Tensor vertex_jacobians, normal_jacobians;
	kernel::WarpedVertexAndNormalJacobians(vertex_jacobians, normal_jacobians, warped_mesh.GetVertexPositions(),
	                                       warped_mesh.GetVertexNormals(), warp_field.GetNodePositions(),
	                                       warp_field.GetNodeRotations(), warp_anchors, warp_anchor_weights);
	return std::make_tuple(vertex_jacobians, normal_jacobians);
}
} // namespace nnrt::rendering::functional
