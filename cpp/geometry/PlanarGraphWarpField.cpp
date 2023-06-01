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
// stdlib includes

// third-party includes

// local includes
#include "geometry/functional/WarpAnchorComputation.h"
#include "geometry/PlanarGraphWarpField.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry {


PlanarGraphWarpField::PlanarGraphWarpField(
		open3d::core::Tensor nodes,
		open3d::core::Tensor edges,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count
) : WarpField(std::move(nodes), node_coverage, threshold_nodes_by_distance_by_default, anchor_count,
              minimum_valid_anchor_count),
    edges(std::move(edges)), edge_weights(std::move(edge_weights)), clusters(std::move(clusters)) {
	auto device = this->node_positions.GetDevice();
	int64_t node_count = this->node_positions.GetLength();
	int64_t max_vertex_degree = this->edges.GetShape(1);

	o3c::AssertTensorDevice(this->edges, device);
	o3c::AssertTensorShape(this->edges, { node_count, max_vertex_degree });
	o3c::AssertTensorDtypes(this->edges, { o3c::Int32, o3c::Int64 });

	if (this->edge_weights.has_value()) {
		o3c::AssertTensorDevice(this->edge_weights.value().get(), device);
		o3c::AssertTensorShape(this->edge_weights.value().get(), { node_count, max_vertex_degree });
		o3c::AssertTensorDtype(this->edge_weights.value().get(), o3c::Float32);
	}

	if (this->clusters.has_value()) {
		o3c::AssertTensorDevice(this->clusters.value().get(), device);
		o3c::AssertTensorShape(this->clusters.value().get(), { node_count });
		o3c::AssertTensorDtype(this->clusters.value().get(), o3c::Int32);
	}

}

//TODO: all these methods should be in GraphWarpField, inherited via CRTP by both PlanarGraphWarpField and GraphWarpField

std::tuple<open3d::core::Tensor, open3d::core::Tensor> PlanarGraphWarpField::PrecomputeAnchorsAndWeights(
		const open3d::t::geometry::TriangleMesh& input_mesh,
		AnchorComputationMethod anchor_computation_method
) const {
	o3c::Tensor anchors, weights;

	if (!input_mesh.HasVertexPositions()) {
		utility::LogError("Input mesh doesn't have vertex positions defined, which are required for computing warp field anchors & weights.");
	}
	const o3c::Tensor& vertex_positions = input_mesh.GetVertexPositions();
	switch (anchor_computation_method) {
		case AnchorComputationMethod::EUCLIDEAN:
			functional::ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
					anchors, weights, vertex_positions, this->node_positions, this->anchor_count,
					this->minimum_valid_anchor_count, this->node_coverage
			);
			break;
		case AnchorComputationMethod::SHORTEST_PATH:
			functional::ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
					anchors, weights, vertex_positions, this->node_positions, this->edges,
					this->anchor_count, this->node_coverage
			);
			break;
		default: utility::LogError("Unsupported AnchorComputationMethod value: {}", anchor_computation_method);
	}

	return std::make_tuple(anchors, weights);
}

PlanarGraphWarpField PlanarGraphWarpField::ApplyTransformations() const {
	return {this->node_positions + this->node_translations, this->edges, this->edge_weights, this->clusters, this->node_coverage,
	        this->threshold_nodes_by_distance_by_default, this->anchor_count, this->minimum_valid_anchor_count};
}

PlanarGraphWarpField::PlanarGraphWarpField(const PlanarGraphWarpField& original, const core::KdTree& index) :
		WarpField(original, index), edges(original.edges.Clone()) {
	if (original.edge_weights.has_value()) {
		auto cloned_edge_weights = original.edge_weights.value().get().Clone();
		this->edge_weights = cloned_edge_weights;
	}
	if (original.clusters.has_value()) {
		auto cloned_clusters = original.clusters.value().get().Clone();
		this->clusters = cloned_clusters;
	}
}

} // namespace nnrt::geometry