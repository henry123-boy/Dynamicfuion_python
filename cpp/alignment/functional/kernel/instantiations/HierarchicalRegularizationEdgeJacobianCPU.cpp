//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/20/23.
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
#include "alignment/functional/kernel/HierarchicalRegularizationEdgeJacobianImpl.h"

namespace nnrt::alignment::functional::kernel {

template
void HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_FixedCoverageWeight<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& edge_jacobians,
		open3d::core::Tensor& node_edge_indices_jagged,
		open3d::core::Tensor& node_edge_counts,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& edge_layer_indices,
		const open3d::core::Tensor& layer_decimation_radii,
		float regularization_weight
);

template
void HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_VariableCoverageWeight<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& edge_jacobians,
		open3d::core::Tensor& node_edges_jagged,
		open3d::core::Tensor& node_edge_counts,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& edges,
		float regularization_weight
);

} // namespace nnrt::alignment::functional::kernel