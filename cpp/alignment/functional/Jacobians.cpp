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
#include "alignment/functional/kernel/Jacobians.h"
#include "Jacobians.h"


namespace utility = open3d::utility;
namespace o3c = open3d::core;

// local includes
namespace nnrt::alignment::functional {
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
WarpedVertexAndNormalJacobians(const open3d::t::geometry::TriangleMesh& canonical_mesh, const geometry::GraphWarpField& warp_field,
                               const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights) {
	if (!canonical_mesh.HasVertexNormals() || !canonical_mesh.HasVertexPositions()) {
		utility::LogError("warped_mesh needs to have both vertex positions and vertex normals defined. In argument, vertex positions are {} defined, "
		                  "and vertex normals are {} defined.", (canonical_mesh.HasVertexPositions() ? "" : "not"),
		                  (canonical_mesh.HasVertexNormals() ? "" : "not"));
	}

	o3c::Tensor warped_vertex_jacobians, warped_normal_jacobians;
	kernel::WarpedVertexAndNormalJacobians(warped_vertex_jacobians, warped_normal_jacobians, canonical_mesh.GetVertexPositions(),
	                                       canonical_mesh.GetVertexNormals(), warp_field.GetNodePositions(),
	                                       warp_field.GetNodeRotations(), warp_anchors, warp_anchor_weights);
	return std::make_tuple(warped_vertex_jacobians, warped_normal_jacobians);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
RenderedVertexAndNormalJacobians(const open3d::t::geometry::TriangleMesh& warped_mesh,
                                 const open3d::core::Tensor& pixel_faces,
                                 const open3d::core::Tensor& barycentric_coordinates,
                                 const open3d::core::Tensor& ndc_intrinsics,
                                 bool perspective_corrected_barycentric_coordinates) {
	if (!warped_mesh.HasVertexNormals() || !warped_mesh.HasVertexPositions() || !warped_mesh.HasTriangleIndices()) {
		utility::LogError("warped_mesh needs to have vertex positions, triangle indices, and vertex normals defined. "
		                  "In argument, vertex positions are {} defined, triangle indices are {} defined, and vertex normals are {} defined.",
		                  (warped_mesh.HasVertexPositions() ? "" : "not"),
		                  (warped_mesh.HasTriangleIndices() ? "" : "not"),
		                  (warped_mesh.HasVertexNormals() ? "" : "not"));
	}

	o3c::Tensor rendered_vertex_jacobians, rendered_normal_jacobians;
	kernel::RenderedVertexAndNormalJacobians(rendered_vertex_jacobians, rendered_normal_jacobians, warped_mesh.GetVertexPositions(),
	                                         warped_mesh.GetTriangleIndices(), warped_mesh.GetVertexNormals(), pixel_faces, barycentric_coordinates,
	                                         ndc_intrinsics, false);

	return std::make_tuple(rendered_vertex_jacobians, rendered_normal_jacobians);
}
} // namespace nnrt::alignment::functional
