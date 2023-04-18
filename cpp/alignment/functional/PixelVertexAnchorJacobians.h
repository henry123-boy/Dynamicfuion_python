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


namespace nnrt::alignment::functional {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
        PixelVertexAnchorJacobiansAndNodeAssociations(
		        const open3d::core::Tensor& rasterized_vertex_position_jacobians,
		        const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
		        const open3d::core::Tensor& warped_vertex_position_jacobians,
		        open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> warped_vertex_normal_jacobians,
		        const open3d::core::Tensor& point_map_vectors,
		        const open3d::core::Tensor& rasterized_normals,
		        const open3d::core::Tensor& residual_mask,
		        const open3d::core::Tensor& pixel_faces,
		        const std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		        const open3d::core::Tensor& face_node_anchor_counts,
		        int64_t node_count,
		        bool use_tukey_penalty,
		        float tukey_penalty_cutoff,
		        nnrt::alignment::IterationMode mode
        );


} // namespace nnrt::alignment::functional