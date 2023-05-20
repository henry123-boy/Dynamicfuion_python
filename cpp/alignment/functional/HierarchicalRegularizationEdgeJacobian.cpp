//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/12/23.
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
// third-party includes

// local includes
#include "alignment/functional/HierarchicalRegularizationEdgeJacobian.h"
#include "alignment/functional/kernel/HierarchicalRegularizationEdgeJacobian.h"

namespace o3c = open3d::core;
namespace nnrt::alignment::functional {
std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
HierarchicalRegularizationEdgeJacobiansAndNodeAssociations(geometry::HierarchicalGraphWarpField& warp_field) {
	const o3c::Tensor& node_positions = warp_field.GetNodePositions(true);
	const o3c::Tensor& node_translations = warp_field.GetNodeTranslations(true);
	const o3c::Tensor& node_rotations = warp_field.GetNodeRotations(true);
	const o3c::Tensor& edges = warp_field.GetEdges();

	o3c::Tensor edge_jacobians, node_edge_jacobian_indices_jagged, node_edge_jacobian_counts;
	kernel::HierarchicalRegularizationEdgeJacobiansAndNodeAssociations(
			edge_jacobians,
			node_edge_jacobian_indices_jagged,
			node_edge_jacobian_counts,
			node_positions,
			node_translations,
			node_rotations,
			edges
	);
	return std::make_tuple(edge_jacobians, node_edge_jacobian_indices_jagged, node_edge_jacobian_counts);
}
} // namespace nnrt::alignment::functional