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
// stdlib includes

// third-party includes

// local includes
#include "alignment/functional/PixelVertexAnchorJacobians.h"
#include "alignment/functional/kernel/PixelVertexAnchorJacobians.h"


namespace nnrt::alignment::functional {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
PixelVertexAnchorJacobiansAndNodeAssociations(
		const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_position_jacobians,
		const open3d::core::Tensor& warped_vertex_normal_jacobians,
		const open3d::core::Tensor& point_map_vectors,
		const open3d::core::Tensor& rasterized_normals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& face_vertices,
		const open3d::core::Tensor& warp_anchors,
		int64_t node_count,
		bool use_tukey_penalty,
		float tukey_penalty_cutoff
) {
	open3d::core::Tensor pixel_jacobians, pixel_node_jacobian_counts, node_pixel_jacobian_indices, node_pixel_jacobian_counts;

	kernel::PixelVertexAnchorJacobiansAndNodeAssociations(
			pixel_jacobians, pixel_node_jacobian_counts, node_pixel_jacobian_indices, node_pixel_jacobian_counts,
			rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians, warped_vertex_position_jacobians,
			warped_vertex_normal_jacobians, point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_vertices, warp_anchors,
			node_count, use_tukey_penalty, tukey_penalty_cutoff
	);

	return std::make_tuple(pixel_jacobians, pixel_node_jacobian_counts, node_pixel_jacobian_indices, node_pixel_jacobian_counts);
}


}//namespace nnrt::alignment::functional