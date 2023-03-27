//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/20/23.
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

// third-party includes
#include <open3d/core/Tensor.h>
#include "alignment/IterationMode.h"

// local includes
namespace nnrt::alignment::functional::kernel {

/**
 * \brief Compute per-pixel jacobians w.r.t. their anchoring node transformations and associate outputs
 * with their anchoring nodes (to be accessed by node)
 * \param pixel_jacobians [out] arrays of per-pixel jacobians, ordered by pixel
 * \param pixel_jacobian_counts [out] total count of jacobians per pixel
 * \param node_pixel_jacobian_indices [out] ordered by node, lists of indices of jacobians in the pixel_jacobians array
 * \param node_pixel_jacobian_counts [out] ordered by node, counts of per-pixel jacobians for each node
 * \param rasterized_vertex_position_jacobians [in] array containing (condensed) jacobians of rasterized vertices w.r.t the warped positions of the triangular face they lie in
 * \param rasterized_vertex_normal_jacobians [in] array containing (condensed) jacobians of rasterized vertices w.r.t the warped vertices & normals of the triangular face they lie in
 * \param warped_vertex_position_jacobians [in] array containing (condensed) jacobians of warped vertex positions w.r.t node transformations, ordered by warped vertex
 * \param warped_vertex_normal_jacobians [in] array containing (condensed) jacobians of warped vertex normals w.r.t node rotations, ordered by warped vertex
 * \param point_map_vectors [in] array containing vectors between rasterized vertices and observed points, ordered by pixel
 * \param rasterized_normals [in] array containing rasterized, per-pixel normals, ordered by pixel
 * \param residual_mask [in] array containing the residual mask -- dictates which pixels to skip during computation
 * \param pixel_faces [in] array containing lists of faces (often, of length one) associated with each pixel
 * \param face_vertices [in] array containing per-face vertex indices
 * \param vertex_anchors [in] array containing vertex anchor indices
 * \param node_count [in] total node count
 */
void PixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_jacobians,
		open3d::core::Tensor& pixel_jacobian_counts,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		const open3d::core::Tensor& warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& face_vertices,
		const open3d::core::Tensor& vertex_anchors,
		int64_t node_count,
		IterationMode mode = IterationMode::ALL,
		bool use_tukey_penalty = false,
		float tukey_penalty_cutoff = 0.01
);

template<open3d::core::Device::DeviceType TDevice>
void PixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_jacobians,
		open3d::core::Tensor& pixel_jacobian_counts,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		const open3d::core::Tensor& warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& face_vertices,
		const open3d::core::Tensor& vertex_anchors,
		int64_t node_count,
		IterationMode mode,
		bool use_tukey_penalty,
		float tukey_penalty_cutoff
);

} // namespace nnrt::alignment::functional::kernel