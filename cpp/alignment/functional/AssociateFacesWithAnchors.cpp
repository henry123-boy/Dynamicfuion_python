//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/17/23.
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
#include "alignment/functional/AssociateFacesWithAnchors.h"
#include "alignment/functional/kernel/AssociateFacesWithAnchors.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;


namespace nnrt::alignment::functional{

std::tuple<std::shared_ptr<open3d::core::Blob>, open3d::core::Tensor>
AssociateFacesWithAnchors(const open3d::core::Tensor& face_vertex_indices, const open3d::core::Tensor& warp_vertex_anchors) {
	std::shared_ptr<open3d::core::Blob> face_node_anchors;
	o3c::Tensor face_anchor_counts;

	kernel::AssociateFacesWithAnchors(face_node_anchors, face_anchor_counts, face_vertex_indices, warp_vertex_anchors);

	return std::make_tuple(face_node_anchors, face_anchor_counts);
}
} // namespace nnrt::alignment::functional
