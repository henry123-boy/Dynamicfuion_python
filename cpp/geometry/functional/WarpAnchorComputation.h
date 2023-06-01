//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/12/22.
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
// stdlib
#include <tuple>

// 3rd party
#include <open3d/core/Tensor.h>
#include <geometry/WarpNodeCoverageComputationMethod.h>


namespace nnrt::geometry::functional {
// region ======================================== EUCLIDEAN =========================================================================================
void ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
);


void ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		int anchor_count,
		int minimum_valid_anchor_count,
		float node_coverage_weight
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		int anchor_count,
		int minimum_valid_anchor_count,
		float node_coverage_weight
);

// endregion
// region ======================================== SHORTEST PATH =====================================================================================
void ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
);

void ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges,
		int anchor_count,
		float node_coverage_weight
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges,
		int anchor_count,
		float node_coverage
);

// endregion

} //namespace nnrt::geometry::functional