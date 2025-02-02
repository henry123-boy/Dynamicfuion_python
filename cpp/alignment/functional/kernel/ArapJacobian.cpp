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
#include "alignment/functional/kernel/ArapJacobian.h"
#include "core/DeviceSelection.h"

namespace nnrt::alignment::functional::kernel {

void ArapEdgeJacobiansAndNodeAssociations_FixedCoverageWeight(
		open3d::core::Tensor& edge_jacobians,

		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& edge_layer_indices,
		const open3d::core::Tensor& layer_decimation_radii,
		float regularization_weight
) {
	core::ExecuteOnDevice(
			node_positions.GetDevice(),
			[&] {
				ArapEdgeJacobiansAndNodeAssociations_FixedCoverageWeight<open3d::core::Device::DeviceType::CPU>(
						edge_jacobians,

						node_positions,
						node_rotations,
						edges,
						edge_layer_indices,
						layer_decimation_radii,
						regularization_weight
				);
			},
			[&] {
				NNRT_IF_CUDA (
						ArapEdgeJacobiansAndNodeAssociations_FixedCoverageWeight<open3d::core::Device::DeviceType::CUDA>(
								edge_jacobians,

								node_positions,
								node_rotations,
								edges, edge_layer_indices,
								layer_decimation_radii,
								regularization_weight
						);
				);
			}
	);
}

void HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_VariableCoverageWeight(
		open3d::core::Tensor&       edge_jacobians,

		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& edges,
		float                       regularization_weight
) {
	core::ExecuteOnDevice(
			node_positions.GetDevice(),
			[&] {
				HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_VariableCoverageWeight<open3d::core::Device::DeviceType::CPU>(
						edge_jacobians,

						node_positions,
						node_coverage_weights,
						node_rotations,
						edges,
						regularization_weight
				);
			},
			[&] {
				NNRT_IF_CUDA (
						HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_VariableCoverageWeight<open3d::core::Device::DeviceType::CUDA>(
								edge_jacobians,

								node_positions,
								node_coverage_weights,
								node_rotations,
								edges,
								regularization_weight
						);
				);
			}
	);
}

} // namespace nnrt::alignment::functional::kernel