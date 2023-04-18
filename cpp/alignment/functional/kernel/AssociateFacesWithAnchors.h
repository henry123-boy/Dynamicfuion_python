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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/core/Blob.h>

// local includes
namespace nnrt::alignment::functional::kernel {

void AssociateFacesWithAnchors(
		std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		open3d::core::Tensor& face_anchor_counts,
		const open3d::core::Tensor& face_vertex_indices,
		const open3d::core::Tensor& warp_vertex_anchors
);

template<open3d::core::Device::DeviceType TDeviceType>
void AssociateFacesWithAnchors(
		std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		open3d::core::Tensor& face_anchor_counts,
		const open3d::core::Tensor& face_vertex_indices,
		const open3d::core::Tensor& warp_vertex_anchors
);

} // namespace nnrt::alignment::functional::kernel