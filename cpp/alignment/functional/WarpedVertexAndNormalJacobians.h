//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/3/23.
//  Copyright (c) 2023 Gregory Kramida
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
#include <tuple>

// third-party includes

// local includes
#include "geometry/GraphWarpField.h"

namespace nnrt::alignment::functional {

/**
 * \brief Compute non-zero, non-trivial entries of warped mesh vertices and normals w.r.t. to node rotations & translations
 * \details Note, the function doesn't compute all the jacobians explicitly.
 * Firstly, the Jacobians w.r.t. rotation it stores in vector form (i.e. zero and negative entries of skew-symmetric
 * form are omitted).
 *
 * Secondly, the vertex Jacobians w.r.t. translation it does not explicitly compute, as these are simply -anchor_weight * identity. Instead,
 * the negative anchor weights (single scalars) are stored to be easily retrieved later for that computation.
 * Hence, these combine into four-component rows of a 2D tensor holding Jacobian information, indexable by vertex index.
 *
 * Finally, the normals' Jacobians w.r.t. translation are all [3x3] zero matrices, so these are not stored at all,
 * keeping the relevant normals' rotation Jacobians as 3-component rows (i.e. positive components of skew-symmetric matrix)
 * of a 2D tensor.
 *
 * \param canonical_mesh canonical (original, not warped) mesh
 * \param warp_field a warp field (typically containing 3D node positions, translation, and rotation information of mesh/scene motion)
 * \param warp_anchors anchors relate canonical (time = 0) vertices to their immediate nodes
 * \param warp_anchor_weights anchor weights constitute how the motion of each node in the warp field affects each vertex
 * \return non-zero, non-trivial entries of warped vertex and normal jacobians w.r.t. to rotations & translations (see details above).
 * vertex position Jacobians are stored in the first tensor in the output tuple, vertex normal Jacobians are stored in the second tensor.
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor> WarpedVertexAndNormalJacobians(
		const open3d::t::geometry::TriangleMesh& canonical_mesh, const geometry::GraphWarpField& warp_field,
		const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights
);

open3d::core::Tensor WarpedVertexRotationJacobians(
		const open3d::t::geometry::TriangleMesh& canonical_mesh, const geometry::GraphWarpField& warp_field,
		const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights
);

} // namespace nnrt::alignment::functional