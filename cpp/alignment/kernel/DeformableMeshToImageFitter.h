//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/5/23.
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

// local includes
#include <open3d/core/Tensor.h>

namespace nnrt::alignment::kernel {

/**
 * \brief Compute per-pixel jacobians w.r.t. their anchoring node transformations and associate outputs
 * with their anchoring nodes (to be accessed by node)
 * \param pixel_jacobians [out] arrays of per-pixel jacobians, ordered by pixel
 * \param pixel_node_jacobian_counts [out] total count of jacobians per pixel
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
void ComputePixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_jacobians,
		open3d::core::Tensor& pixel_node_jacobian_counts,
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
		bool use_tukey_penalty = false,
		float tukey_penalty_cutoff = 0.01
);

template<open3d::core::Device::DeviceType TDevice, bool TUseTukeyPenalty = false>
void ComputePixelVertexAnchorJacobiansAndNodeAssociations(
		open3d::core::Tensor& pixel_jacobians,
		open3d::core::Tensor& pixel_node_jacobian_counts,
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
		float tukey_penalty_cutoff_cm = 0.01
);

/**
 * \brief Comes up with a compact, ordered sparse representation of jacobian J of residuals w.r.t. node motion deltas, easily accessible with
 * [node index, pixel index] (in that order).
 *
 * \param node_jacobians [out]
 * \param node_jacobian_ranges [out]
 * \param node_jacobian_pixel_indices [out]
 * \param node_pixel_jacobian_indices_jagged [in, out] Shape: [node_count x MAX_JACOBAINS_PER_NODE]; a tensor holding a "jagged" 2d array of int32 indices,
 * with each row containing the indices of jacobians associated with a single node, and rows ordered by node index.
 * \param node_pixel_counts [in] Shape: [node_count], a 1d array of actual counts of jacobians for each node
 * \param pixel_jacobians [in] Shape: [pixel_count x vertices_in_face(3) x (max) anchor_count(per vertex) x 6 (3 for node rotation update + 3 for node translation update];
 * a "jagged" 2d array of derivatives, where each row is associated with a single output pixel. A row's cells are the derivative of the residual at
 * the row's pixel with respect to a single node (that affects the given pixel, if at all) -- not all cells in each rows are expected to have valid data,
 * so long as the node_pixel_jacobian_indices_jagged argument contains indices pointing only to valid entries.
 */
void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
		open3d::core::Tensor& node_jacobians, // out
		open3d::core::Tensor& node_jacobian_ranges, // out
		open3d::core::Tensor& node_jacobian_pixel_indices, // out
		open3d::core::Tensor& node_pixel_jacobian_indices_jagged, //[in, out]
		const open3d::core::Tensor& node_pixel_counts,
		const open3d::core::Tensor& pixel_jacobians
);

template<open3d::core::Device::DeviceType TDevice>
void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
		open3d::core::Tensor& node_jacobians,
		open3d::core::Tensor& node_jacobian_ranges,
		open3d::core::Tensor& node_jacobian_pixel_indices,
		open3d::core::Tensor& node_pixel_jacobian_indices_jagged,
		const open3d::core::Tensor& node_pixel_counts,
		const open3d::core::Tensor& pixel_jacobians
);

void ComputeHessianApproximationBlocks_UnorderedNodePixels(
		open3d::core::Tensor& workload_index,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts
);

template<open3d::core::Device::DeviceType TDevice>
void ComputeHessianApproximationBlocks_UnorderedNodePixels(
		open3d::core::Tensor& workload_index,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts
);

void ComputeNegativeGradient_UnorderedNodePixels(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex
);

template<open3d::core::Device::DeviceType TDevice>
void ComputeNegativeGradient_UnorderedNodePixels(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex
);

} // namespace nnrt::alignment::kernel