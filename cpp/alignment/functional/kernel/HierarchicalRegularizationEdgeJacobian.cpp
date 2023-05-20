//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/18/23.
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
#include <open3d/core/Tensor.h>

// local includes
#include "alignment/functional/kernel/HierarchicalRegularizationEdgeJacobian.h"
#include "core/DeviceSelection.h"

namespace nnrt::alignment::functional::kernel {

void HierarchicalRegularizationEdgeJacobiansAndNodeAssociations(
		open3d::core::Tensor& edge_jacobians,
		open3d::core::Tensor& node_edge_jacobian_indices_jagged,
		open3d::core::Tensor& node_edge_jacobian_counts,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& edges
) {
	core::ExecuteOnDevice(
			node_positions.GetDevice(),
			[&] {
				HierarchicalRegularizationEdgeJacobiansAndNodeAssociations<open3d::core::Device::DeviceType::CPU>(
						edge_jacobians,
						node_edge_jacobian_indices_jagged,
						node_edge_jacobian_counts,
						node_positions,
						node_rotations,
						node_translations,
						edges
				);
			},
			[&] {
				NNRT_IF_CUDA (HierarchicalRegularizationEdgeJacobiansAndNodeAssociations<open3d::core::Device::DeviceType::CUDA>(
						edge_jacobians,
						node_edge_jacobian_indices_jagged,
						node_edge_jacobian_counts,
						node_positions,
						node_rotations,
						node_translations,
						edges
				); );
			}

}

} // namespace nnrt::alignment::functional::kernel