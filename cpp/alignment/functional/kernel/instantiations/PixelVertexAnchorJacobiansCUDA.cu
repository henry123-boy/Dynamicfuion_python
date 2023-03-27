//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/27/23.
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
#include "alignment/functional/kernel/PixelVertexAnchorJacobiansImpl.h"

namespace nnrt::alignment::functional::kernel {

template
void PixelVertexAnchorJacobiansAndNodeAssociations<open3d::core::Device::DeviceType::CUDA>(
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