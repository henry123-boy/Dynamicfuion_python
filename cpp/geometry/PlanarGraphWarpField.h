//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/16/23.
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

// local includes
#include "geometry/WarpField.h"
#include "geometry/functional/AnchorComputationMethod.h"

namespace nnrt::geometry {


class PlanarGraphWarpField : public WarpField {
public:
	PlanarGraphWarpField(
			open3d::core::Tensor nodes,
			open3d::core::Tensor edges,
			open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights,
			open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0
	);
	PlanarGraphWarpField(const PlanarGraphWarpField& original) = default;
	PlanarGraphWarpField(PlanarGraphWarpField&& other) = default;

	std::tuple<open3d::core::Tensor, open3d::core::Tensor> PrecomputeAnchorsAndWeights(
			const open3d::t::geometry::TriangleMesh& input_mesh,
			AnchorComputationMethod anchor_computation_method
	) const;

	const open3d::core::Tensor edges;
	open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights;
	open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters;

	PlanarGraphWarpField ApplyTransformations() const;

protected:
	PlanarGraphWarpField(const PlanarGraphWarpField& original, const core::KdTree& index);

};


} // namespace nnrt::geometry