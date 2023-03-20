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
#include "alignment/functional/kernel/RasterizedSurfaceJacobians.h"
#include "alignment/functional/RasterizedSurfaceJacobians.h"


namespace utility = open3d::utility;
namespace o3c = open3d::core;

// local includes
namespace nnrt::alignment::functional {
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
RasterizedSurfaceJacobians(const open3d::t::geometry::TriangleMesh& warped_mesh,
                           const open3d::core::Tensor& pixel_faces,
                           const open3d::core::Tensor& barycentric_coordinates,
                           const open3d::core::Tensor& ndc_intrinsic_matrix,
                           bool use_perspective_corrected_barycentric_coordinates) {
	if (!warped_mesh.HasVertexNormals() || !warped_mesh.HasVertexPositions() || !warped_mesh.HasTriangleIndices()) {
		utility::LogError("warped_mesh needs to have vertex positions, triangle indices, and vertex normals defined. "
		                  "In argument, vertex positions are {} defined, triangle indices are {} defined, and vertex normals are {} defined.",
		                  (warped_mesh.HasVertexPositions() ? "" : "not"),
		                  (warped_mesh.HasTriangleIndices() ? "" : "not"),
		                  (warped_mesh.HasVertexNormals() ? "" : "not"));
	}

	o3c::Tensor rendered_vertex_jacobians, rendered_normal_jacobians;
	kernel::RasterizedSurfaceJacobians(rendered_vertex_jacobians, rendered_normal_jacobians,
	                                   warped_mesh.GetVertexPositions(),
	                                   warped_mesh.GetTriangleIndices(),
	                                   warped_mesh.GetVertexNormals(),
	                                   pixel_faces, barycentric_coordinates,
	                                   ndc_intrinsic_matrix, use_perspective_corrected_barycentric_coordinates, true);

	return std::make_tuple(rendered_vertex_jacobians, rendered_normal_jacobians);
}

open3d::core::Tensor RasterizedVertexJacobians(
		const open3d::t::geometry::TriangleMesh& warped_mesh,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& barycentric_coordinates,
		const open3d::core::Tensor& ndc_intrinsic_matrix,
		bool use_perspective_corrected_barycentric_coordinates
) {
	if (!warped_mesh.HasVertexPositions() || !warped_mesh.HasTriangleIndices()) {
		utility::LogError("warped_mesh needs to have vertex positions and triangle indices defined. "
		                  "In argument, vertex positions are {} defined and triangle indices are {} defined.",
		                  (warped_mesh.HasVertexPositions() ? "" : "not"),
		                  (warped_mesh.HasTriangleIndices() ? "" : "not"));
	}

	o3c::Tensor rendered_vertex_jacobians;
	kernel::RasterizedSurfaceJacobians(rendered_vertex_jacobians, utility::nullopt,
	                                   warped_mesh.GetVertexPositions(),
	                                   warped_mesh.GetTriangleIndices(),
	                                   warped_mesh.GetVertexNormals(),
	                                   pixel_faces, barycentric_coordinates,
	                                   ndc_intrinsic_matrix, use_perspective_corrected_barycentric_coordinates, true);

	return rendered_vertex_jacobians;
}

} // namespace nnrt::alignment::functional
