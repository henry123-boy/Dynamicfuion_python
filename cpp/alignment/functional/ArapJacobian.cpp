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
#include "alignment/functional/ArapJacobian.h"
#include "alignment/functional/kernel/ArapJacobian.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace nnrt::alignment::functional {
open3d::core::Tensor
ComputeDenseArapEdgeJacobians(
		geometry::HierarchicalGraphWarpField& warp_field,
		float regularization_weight
) {
	const o3c::Tensor& node_positions = warp_field.GetNodePositions(true);
	const o3c::Tensor& node_rotations = warp_field.GetNodeRotations(true);
	const o3c::Tensor& edges = warp_field.GetEdges();

	o3c::Tensor edge_jacobians;

	switch(warp_field.warp_node_coverage_computation_method){
		case geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE:{
			const o3c::Tensor& edge_layer_indices = warp_field.GetEdgeLayerIndices();
			const o3c::Tensor& layer_decimation_radii = warp_field.GetLayerDecimationRadii();
			kernel::ArapEdgeJacobiansAndNodeAssociations_FixedCoverageWeight(
					edge_jacobians,

					node_positions,
					node_rotations,
					edges,
					edge_layer_indices,
					layer_decimation_radii,
					regularization_weight
			);
		}
			break;

		case geometry::WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE:{
			const o3c::Tensor& node_coverage_weights = warp_field.GetNodeCoverageWeights(true);
			kernel::HierarchicalRegularizationEdgeJacobiansAndNodeAssociations_VariableCoverageWeight(
					edge_jacobians,

					node_positions,
					node_coverage_weights,
					node_rotations,
					edges,
					regularization_weight
			);
		}
			break;
		default:
			utility::LogError("Unsupported warp node coverage computation method: {}", warp_field.warp_node_coverage_computation_method);
			break;
	}

	return edge_jacobians;
}
} // namespace nnrt::alignment::functional