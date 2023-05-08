//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/21.
//  Copyright (c) 2021 Gregory Kramida
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
// stdlib
#include <utility>

// local
#include "geometry/GraphWarpField.h"
#include "core/linalg/Matmul3D.h"
#include "geometry/functional/WarpAnchorComputation.h"
#include "geometry/functional/GeometrySampling.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::geometry {


WarpField::WarpField(
		o3c::Tensor nodes,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count,
		int layer_count,
		float decimation_radius
) :
		nodes(std::move(nodes)), node_coverage(node_coverage), anchor_count(anchor_count),
		threshold_nodes_by_distance_by_default(threshold_nodes_by_distance_by_default),
		minimum_valid_anchor_count(minimum_valid_anchor_count),

		index(this->nodes), node_coverage_squared(node_coverage * node_coverage),
		kd_tree_nodes(this->index.GetNodes()), kd_tree_node_count(this->index.GetNodeCount()),
		node_indexer(this->nodes, 1),

		rotations({this->nodes.GetLength(), 3, 3}, o3c::Dtype::Float32, this->nodes.GetDevice()),
		translations(o3c::Tensor::Zeros({this->nodes.GetLength(), 3}, o3c::Dtype::Float32, this->nodes.GetDevice())),
		rotations_data(this->rotations.GetDataPtr<float>()), translations_data(this->translations.GetDataPtr<float>()) {

	int64_t node_count = this->nodes.GetLength();
	o3c::AssertTensorShape(this->nodes, {node_count, 3});
	o3c::AssertTensorDtype(this->nodes, o3c::Float32);

	this->ResetRotations();
	this->BuildRegularizationLayers(layer_count, decimation_radius);
}

WarpField::WarpField(const WarpField& original, const core::KdTree& index) :
		nodes(index.GetPoints()),
		node_coverage(original.node_coverage), anchor_count(original.anchor_count),
		threshold_nodes_by_distance_by_default(original.threshold_nodes_by_distance_by_default),
		minimum_valid_anchor_count(original.minimum_valid_anchor_count),

		index(index),
		node_coverage_squared(original.node_coverage_squared),
		kd_tree_nodes(this->index.GetNodes()), kd_tree_node_count(this->index.GetNodeCount()),
		node_indexer(this->nodes, 1),

		rotations(original.rotations.Clone()),
		translations(original.translations.Clone()),
		rotations_data(this->rotations.GetDataPtr<float>()),
		translations_data(this->translations.GetDataPtr<float>()) {}

o3c::Tensor WarpField::GetWarpedNodes() const {
	return nodes + this->translations;
}

o3c::Tensor WarpField::GetNodeExtent() const {
	o3c::Tensor minMax({2, 3}, nodes.GetDtype(), nodes.GetDevice());
	minMax.Slice(0, 0, 1) = nodes.Min({0});
	minMax.Slice(0, 1, 2) = nodes.Max({0});
	return minMax;
}

open3d::t::geometry::TriangleMesh
WarpField::WarpMesh(
		const open3d::t::geometry::TriangleMesh& input_mesh, bool disable_neighbor_thresholding,
		const open3d::core::Tensor& extrinsics/* = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) const {
	if (disable_neighbor_thresholding) {
		return functional::WarpTriangleMesh(input_mesh, this->nodes, this->rotations, this->translations, this->anchor_count, this->node_coverage,
		                                    false, 0, extrinsics);
	} else {
		return functional::WarpTriangleMesh(input_mesh, this->nodes, this->rotations, this->translations, this->anchor_count, this->node_coverage,
		                                    this->threshold_nodes_by_distance_by_default, this->minimum_valid_anchor_count, extrinsics);
	}

}

open3d::t::geometry::TriangleMesh
WarpField::WarpMesh(
		const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& anchors,
		const open3d::core::Tensor& weights, bool disable_neighbor_thresholding,
		const open3d::core::Tensor& extrinsics/* = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) const {
	if (disable_neighbor_thresholding) {
		return functional::WarpTriangleMeshUsingSuppliedAnchors(input_mesh, this->nodes, this->rotations, this->translations, anchors, weights, false,
		                                                        0, extrinsics);
	} else {
		return functional::WarpTriangleMeshUsingSuppliedAnchors(input_mesh, this->nodes, this->rotations, this->translations, anchors, weights,
		                                                        this->threshold_nodes_by_distance_by_default, this->minimum_valid_anchor_count,
		                                                        extrinsics);
	}
}


const core::KdTree& WarpField::GetIndex() const {
	return this->index;
}

void WarpField::ResetRotations() {
	for (int i_node = 0; i_node < this->nodes.GetLength(); i_node++) {
		rotations.Slice(0, i_node, i_node + 1) = o3c::Tensor::Eye(3, o3c::Dtype::Float32, this->nodes.GetDevice());
	}
}

WarpField WarpField::ApplyTransformations() const {
	return {this->nodes + this->translations, this->node_coverage,
	        this->threshold_nodes_by_distance_by_default, this->anchor_count, this->minimum_valid_anchor_count};
}

WarpField WarpField::Clone() {
	return {*this, this->index.Clone()};
}

void WarpField::SetNodeRotations(const o3c::Tensor& node_rotations) {
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorShape(node_rotations, { this->nodes.GetLength(), 3, 3 });
	this->rotations = node_rotations;
	rotations_data = this->rotations.GetDataPtr<float>();
}

void WarpField::SetNodeTranslations(const o3c::Tensor& node_translations) {
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorShape(node_translations, { this->nodes.GetLength(), 3 });
	this->translations = node_translations;
	translations_data = this->translations.GetDataPtr<float>();
}

void WarpField::TranslateNodes(const o3c::Tensor& node_translation_deltas) {
	this->translations += node_translation_deltas;
}

void WarpField::RotateNodes(const o3c::Tensor& node_rotation_deltas) {
	o3c::Tensor new_rotations;
	core::linalg::Matmul3D(new_rotations, this->rotations, node_rotation_deltas);
	this->rotations = new_rotations;
	rotations_data = this->rotations.GetDataPtr<float>();
}

open3d::core::Tensor WarpField::GetNodeRotations() {
	return this->rotations;
}

const open3d::core::Tensor& WarpField::GetNodeRotations() const {
	return rotations;
}

open3d::core::Tensor WarpField::GetNodeTranslations() {
	return this->translations;
}

const open3d::core::Tensor& WarpField::GetNodeTranslations() const {
	return this->translations;
}

const open3d::core::Tensor& WarpField::GetNodePositions() const {
	return nodes;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor> WarpField::PrecomputeAnchorsAndWeights(
		const open3d::t::geometry::TriangleMesh& input_mesh
) const {
	o3c::Tensor anchors, weights;

	if (!input_mesh.HasVertexPositions()) {
		utility::LogError("Input mesh doesn't have vertex positions defined, which are required for computing warp field anchors & weights.");
	}
	const o3c::Tensor& vertex_positions = input_mesh.GetVertexPositions();
	functional::ComputeAnchorsAndWeightsEuclidean(
			anchors, weights, vertex_positions, this->nodes, this->anchor_count,
			this->minimum_valid_anchor_count, this->node_coverage
	);
	return std::make_tuple(anchors, weights);
}

const GraphWarpFieldRegularizationLayer& WarpField::GetRegularizationLevel(int i_layer) const {
	return this->regularization_layers[i_layer];
}

int WarpField::GetRegularizationLevelCount() const {
	return static_cast<int>(this->regularization_layers.size());
}

void WarpField::BuildRegularizationLayers(int count, float decimation_radius) {
	this->regularization_layers = {};
	o3c::Tensor empty_tensor;
	this->regularization_layers.emplace_back(GraphWarpFieldRegularizationLayer{decimation_radius, this->nodes, empty_tensor});

	float current_decimation_radius = decimation_radius;
	for (int i_layer = 1; i_layer < count; i_layer++) {
		o3c::Tensor layer_nodes = geometry::functional::FastMeanRadiusDownsample3dPoints(this->nodes, current_decimation_radius);
		current_decimation_radius = (static_cast<float>(i_layer) + 1) * decimation_radius;
		this->regularization_layers.emplace_back(GraphWarpFieldRegularizationLayer{current_decimation_radius, layer_nodes, empty_tensor});
	}
	if (count > 1) {
		o3c::Tensor edges_layer_0, squared_distances;
		this->index.FindKNearestToPoints(edges_layer_0, squared_distances, this->regularization_layers[1].nodes, 4);
		//TODO: add edges for other layers
	}
}


PlanarGraphWarpField::PlanarGraphWarpField(
		open3d::core::Tensor nodes,
		open3d::core::Tensor edges,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count,
		int layer_count,
		float decimation_radius
) : WarpField(std::move(nodes), node_coverage, threshold_nodes_by_distance_by_default, anchor_count,
              minimum_valid_anchor_count, layer_count, decimation_radius),
    edges(std::move(edges)), edge_weights(std::move(edge_weights)), clusters(std::move(clusters)) {
	auto device = this->nodes.GetDevice();
	int64_t node_count = this->nodes.GetLength();
	int64_t max_vertex_degree = this->edges.GetShape(1);

	o3c::AssertTensorDevice(this->edges, device);
	o3c::AssertTensorShape(this->edges, { node_count, max_vertex_degree });
	o3c::AssertTensorDtypes(this->edges, { o3c::Int32, o3c::Int64});

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
			functional::ComputeAnchorsAndWeightsEuclidean(
					anchors, weights, vertex_positions, this->nodes, this->anchor_count,
					this->minimum_valid_anchor_count, this->node_coverage
			);
			break;
		case AnchorComputationMethod::SHORTEST_PATH:
			functional::ComputeAnchorsAndWeightsShortestPath(
					anchors, weights, vertex_positions, this->nodes, this->edges,
					this->anchor_count, this->node_coverage
			);
			break;
		default: utility::LogError("Unsupported AnchorComputationMethod value: {}", anchor_computation_method);
	}

	return std::make_tuple(anchors, weights);
}

PlanarGraphWarpField PlanarGraphWarpField::ApplyTransformations() const {
	return {this->nodes + this->translations, this->edges, this->edge_weights, this->clusters, this->node_coverage,
	        this->threshold_nodes_by_distance_by_default, this->anchor_count, this->minimum_valid_anchor_count};
}

PlanarGraphWarpField::PlanarGraphWarpField(const PlanarGraphWarpField& original, const core::KdTree& index):
		WarpField(original, index), edges(original.edges.Clone())
{
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