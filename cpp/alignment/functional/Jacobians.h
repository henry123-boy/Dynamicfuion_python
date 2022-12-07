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
#include "geometry/GraphWarpField.h"

namespace nnrt::alignment::functional {
/**
 * \brief Compute non-zero, non-trivial entries of warped mesh vertices and normals w.r.t. to node rotations & translations
 * \details Note, the function doesn't compute all the jacobians explicitly.
 * Firstly, the Jacobians w.r.t. rotation it stores in vector form (i.e. zero entries of skew-symmetric form are omitted).
 * Secondly, the vertex Jacobians w.r.t. translation it does not explicitly compute, as these are simply -anchor_weight * identity. Instead,
 * the negative anchor weights are stored to be easily retrieved later for that computation.
 * Finally, the normals' Jacobians w.r.t. translation are all [3x3] zero matrices, so these are not stored at all.
 * \param canonical_mesh canonical (original, not warped) mesh
 * \param warp_field
 * \param warp_anchors
 * \param warp_anchor_weights
 * \return non-zero, non-trivial entries of warped vertex and normal jacobians w.r.t. to rotations & translations (see details above).
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor> WarpedVertexAndNormalJacobians(
	const open3d::t::geometry::TriangleMesh& canonical_mesh, const geometry::GraphWarpField& warp_field,
	const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor> RenderedVertexAndNormalJacobians(
	const open3d::t::geometry::TriangleMesh& warped_mesh, const open3d::core::Tensor& pixel_faces,
	const open3d::core::Tensor& barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsics,
	bool perspective_corrected_barycentric_coordinates
);


} // namespace nnrt::alignment::functional