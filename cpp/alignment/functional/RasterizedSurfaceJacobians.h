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

// local includes
#include "geometry/HierarchicalGraphWarpField.h"

namespace nnrt::alignment::functional {
/**
 * \brief Compute non-zero, non-trivial entries of the Jacobians of rendered vertices and normals with respect to
 * (warped) geometry
 * \details
 * "Rendered vertices" in this context refers to intersections of rays through each "pixel" of a virtual camera with
 * mesh faces (that are closest to the focal point).
 * The position of each rendered vertex is entirely independent from the normals of the vertices defining the face,
 * hence the Jacobians of the rendered vertex positions' are expressed as 3x9 matrices, indexable by pixel, i.e.
 * the output tensor has dimensions image_height x image_width x 3 x 9.
 *
 * The normal at each ray intersection does depend on actual (warped) geometric vertex positions as well as vertex
 * normals (being interpolated between three vertices of the face where the intersection occurs). However, its Jacobian
 * with respect to the three normals can be expressed as the Kronecker product of the barycentric coordinates with a
 * 3x3 identity matrix. Hence, we store the barycentric coordinates as a 1 x 3 vector appended to the 3 x 9 Jacobian with
 * respect to vertex positions. The entire normals' jacobian is then output as a tensor with the following dimensions:
 * image_height x image_width x 3 x 10. Caveat: storage order is such that the 3x9 Jacobian occupies the first 27 entries
 * of each 3x10 block, and the remaining 3 entries contain the barycentric coordinates, i.e. the 3 entries at the end of
 * the last row, not the last 3-entry column.
 *
 * \param warped_mesh mesh (assumed warped if this is used for warped mesh fitting)
 * \param pixel_faces list of faces associated with each pixel (usually, association is determined by ray intersection).
 * Dimensions: height X width X face_count_per_pixel.
 * Note: only the first face in each list will be utilized for Jacobian computation.
 * \param barycentric_coordinates barycentric coordinates associated with each face and pixel ray
 * Dimensions: height X width X face_count_per_pixel x 3.
 * * Note: only the first set of coordinates in each list will be utilized for Jacobian computation.
 * \param ndc_intrinsic_matrix intrinsic matrix that projects from 3D camera space to normalized device coordinates
 * \param use_perspective_corrected_barycentric_coordinates whether or not to assume & use perspective correction during computations
 * \return non-zero, non-trivial entries of the Jacobians of rendered vertices and normals with respect to
 * (warped) geometry (see details). Rendered vertex jacobians are stored in the first output tensor, rendered
 * normal jacobians are stored in the second one.
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor> RasterizedSurfaceJacobians(
	const open3d::t::geometry::TriangleMesh& warped_mesh, const open3d::core::Tensor& pixel_faces,
	const open3d::core::Tensor& barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsic_matrix,
	bool use_perspective_corrected_barycentric_coordinates
);

open3d::core::Tensor RasterizedVertexJacobians(
		const open3d::t::geometry::TriangleMesh& warped_mesh, const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsic_matrix,
		bool use_perspective_corrected_barycentric_coordinates
);



} // namespace nnrt::alignment::functional