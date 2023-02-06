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
		int64_t node_count
);

template<open3d::core::Device::DeviceType TDevice>
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
		int64_t node_count
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