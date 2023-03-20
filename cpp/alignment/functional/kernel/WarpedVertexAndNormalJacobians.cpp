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
// stdlib includes

// third-party includes

// local includes
#include "core/DeviceSelection.h"
#include "WarpedVertexAndNormalJacobians.h"

namespace nnrt::alignment::functional::kernel {

void WarpedVertexAndNormalJacobians(
		open3d::core::Tensor& vertex_jacobians,
		open3d::core::Tensor& normal_jacobians,
		const open3d::core::Tensor& vertex_positions,
		const open3d::core::Tensor& vertex_normals,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& warp_anchors,
		const open3d::core::Tensor& warp_anchor_weights,
		bool store_anchor_weights_for_translation_jacobians
) {
	if (store_anchor_weights_for_translation_jacobians) {
		core::ExecuteOnDevice(
				vertex_positions.GetDevice(),
				[&] {
					WarpedVertexAndNormalJacobians<open3d::core::Device::DeviceType::CPU, true>(
							vertex_jacobians, normal_jacobians, vertex_positions, vertex_normals, node_positions, node_rotations, warp_anchors,
							warp_anchor_weights);
				},
				[&] {
					NNRT_IF_CUDA(
							WarpedVertexAndNormalJacobians<open3d::core::Device::DeviceType::CUDA, true>(
									vertex_jacobians, normal_jacobians, vertex_positions, vertex_normals, node_positions, node_rotations,
									warp_anchors,
									warp_anchor_weights);
					);
				}
		);
	} else {
		core::ExecuteOnDevice(
				vertex_positions.GetDevice(),
				[&] {
					WarpedVertexAndNormalJacobians<open3d::core::Device::DeviceType::CPU, false>(
							vertex_jacobians, normal_jacobians, vertex_positions, vertex_normals, node_positions, node_rotations, warp_anchors,
							warp_anchor_weights);
				},
				[&] {
					NNRT_IF_CUDA(
							WarpedVertexAndNormalJacobians<open3d::core::Device::DeviceType::CUDA, false>(
									vertex_jacobians, normal_jacobians, vertex_positions, vertex_normals, node_positions, node_rotations,
									warp_anchors,
									warp_anchor_weights);
					);
				}
		);
	}


}

} // namespace nnrt::alignment::functional::kernel