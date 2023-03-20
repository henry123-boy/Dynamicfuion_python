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
#include <open3d/utility/Optional.h>

// local includes

namespace nnrt::alignment::functional::kernel {

void WarpedVertexAndNormalJacobians(
		open3d::core::Tensor& vertex_position_jacobians,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> vertex_normal_jacobians,
		const open3d::core::Tensor& vertex_positions,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> vertex_normals,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& warp_anchors,
		const open3d::core::Tensor& warp_anchor_weights,
		bool vertex_rotation_only
);

template<open3d::core::Device::DeviceType TDeviceType, bool TVertexRotationOnly = false>
void WarpedVertexAndNormalJacobians(
		open3d::core::Tensor& vertex_position_jacobians,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> vertex_normal_jacobians,
		const open3d::core::Tensor& vertex_positions,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& vertex_normals,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& warp_anchors,
		const open3d::core::Tensor& warp_anchor_weights
);


} // namespace nnrt::alignment::functional::kernel