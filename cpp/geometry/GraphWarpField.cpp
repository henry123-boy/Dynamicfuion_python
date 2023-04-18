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

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::geometry {


GraphWarpField::GraphWarpField(
		o3c::Tensor nodes,
		o3c::Tensor edges,
		utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights,
		utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count
) :
		nodes(std::move(nodes)), edges(std::move(edges)), edge_weights(std::move(edge_weights)), clusters(std::move(clusters)),
		node_coverage(node_coverage), anchor_count(anchor_count), threshold_nodes_by_distance_by_default(threshold_nodes_by_distance_by_default),
		minimum_valid_anchor_count(minimum_valid_anchor_count),

		index(this->nodes), node_coverage_squared(node_coverage * node_coverage),
		kd_tree_nodes(this->index.GetNodes()), kd_tree_node_count(this->index.GetNodeCount()),
		node_indexer(this->nodes, 1),

		rotations({this->nodes.GetLength(), 3, 3}, o3c::Dtype::Float32, this->nodes.GetDevice()),
		translations(o3c::Tensor::Zeros({this->nodes.GetLength(), 3}, o3c::Dtype::Float32, this->nodes.GetDevice())),
		rotations_data(this->rotations.GetDataPtr<float>()), translations_data(this->translations.GetDataPtr<float>()) {
	auto device = this->nodes.GetDevice();
	o3c::AssertTensorDevice(this->edges, device);

	auto nodes_shape = this->nodes.GetShape();
	auto edges_shape = this->edges.GetShape();


	if (nodes_shape.size() != 2 || edges_shape.size() != 2) {
		utility::LogError("Arguments `nodes` and `edges` need to have two dimensions. Got dimension counts {} and {}, respectively.",
		                  nodes_shape.size(), edges_shape.size());
	}
	const int64_t node_count = nodes_shape[0];
	if (nodes_shape[1] != 3) {
		utility::LogError("Argument nodes needs to have size N x 3, has size N x {}.", nodes_shape[1]);
	}
	if (edges_shape[0] != node_count) {
		utility::LogError("Argument `edges_shape` needs to have shape ({}, X), where first dimension is the node count N"
		                  " and the second is the edge degree X, but has shape {}", node_count, edges_shape);
	}

	if (this->edge_weights.has_value()) {
		o3c::AssertTensorDevice(this->edge_weights.value().get(), device);
		auto edge_weights_shape = this->edge_weights.value().get().GetShape();
		if (edge_weights_shape.size() != 2) {
			utility::LogError("If a tensor is provided in `edge_weights`, it needs to have two dimensions. Got dimension count {}.",
			                  edge_weights_shape.size());
		}
		if (edge_weights_shape != edges_shape) {
			utility::LogError("Tensors `edges` & `edge_weights` need to have the same shape. Got shapes: {} and {}, respectively.", edges_shape,
			                  edge_weights_shape);
		}
	}

	if (this->clusters.has_value()) {
		o3c::AssertTensorDevice(this->clusters.value().get(), device);
		auto clusters_shape = this->clusters.value().get().GetShape();
		if (clusters_shape.size() != 1) {
			utility::LogError("If a tensor is provided in `clusters`, it needs to have one dimension. Got dimension count {}.",
			                  clusters_shape.size());
		}
		if (clusters_shape[0] != node_count) {
			utility::LogError("If a tensor is provided in `clusters`, `clusters` needs to be a vector of the size {} (node count); got size {}.",
			                  node_count, clusters_shape[0]);
		}
	}


	this->ResetRotations();
}

GraphWarpField::GraphWarpField(const GraphWarpField& original, const core::KdTree& index) :
		nodes(index.GetPoints()), edges(original.edges.Clone()),
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
		translations_data(this->translations.GetDataPtr<float>()) {
	if (original.edge_weights.has_value()) {
		//TODO: not sure this actually works on memory level, verify
		auto cloned_edge_weights = original.edge_weights.value().get().Clone();
		this->edge_weights = cloned_edge_weights;
	}
	if (original.clusters.has_value()) {
		//TODO: not sure this actually works on memory level, verify
		auto cloned_clusters = original.clusters.value().get().Clone();
		this->clusters = cloned_clusters;
	}
}

o3c::Tensor GraphWarpField::GetWarpedNodes() const {
	return nodes + this->translations;
}

o3c::Tensor GraphWarpField::GetNodeExtent() const {
	o3c::Tensor minMax({2, 3}, nodes.GetDtype(), nodes.GetDevice());
	minMax.Slice(0, 0, 1) = nodes.Min({0});
	minMax.Slice(0, 1, 2) = nodes.Max({0});
	return minMax;
}

open3d::t::geometry::TriangleMesh
GraphWarpField::WarpMesh(
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
GraphWarpField::WarpMesh(
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


const core::KdTree& GraphWarpField::GetIndex() const {
	return this->index;
}

void GraphWarpField::ResetRotations() {
	for (int i_node = 0; i_node < this->nodes.GetLength(); i_node++) {
		rotations.Slice(0, i_node, i_node + 1) = o3c::Tensor::Eye(3, o3c::Dtype::Float32, this->nodes.GetDevice());
	}
}

GraphWarpField GraphWarpField::ApplyTransformations() const {
	return {this->nodes + this->translations, this->edges, this->edge_weights, this->clusters, this->node_coverage,
	        this->threshold_nodes_by_distance_by_default, this->anchor_count, this->minimum_valid_anchor_count};
}

GraphWarpField GraphWarpField::Clone() {
	return {*this, this->index.Clone()};
}

void GraphWarpField::SetNodeRotations(const o3c::Tensor& node_rotations) {
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorShape(node_rotations, { this->nodes.GetLength(), 3, 3 });
	this->rotations = node_rotations;
	rotations_data = this->rotations.GetDataPtr<float>();
}

void GraphWarpField::SetNodeTranslations(const o3c::Tensor& node_translations) {
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorShape(node_translations, { this->nodes.GetLength(), 3 });
	this->translations = node_translations;
	translations_data = this->translations.GetDataPtr<float>();
}

void GraphWarpField::TranslateNodes(const o3c::Tensor& node_translation_deltas) {
	this->translations += node_translation_deltas;
}

void GraphWarpField::RotateNodes(const o3c::Tensor& node_rotation_deltas) {
	o3c::Tensor new_rotations;
	core::linalg::Matmul3D(new_rotations, this->rotations, node_rotation_deltas);
	this->rotations = new_rotations;
	rotations_data = this->rotations.GetDataPtr<float>();
}

open3d::core::Tensor GraphWarpField::GetNodeRotations() {
	return this->rotations;
}

const open3d::core::Tensor& GraphWarpField::GetNodeRotations() const {
	return rotations;
}

open3d::core::Tensor GraphWarpField::GetNodeTranslations() {
	return this->translations;
}

const open3d::core::Tensor& GraphWarpField::GetNodeTranslations() const {
	return this->translations;
}

const open3d::core::Tensor& GraphWarpField::GetNodePositions() const {
	return nodes;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor> GraphWarpField::PrecomputeAnchorsAndWeights(
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

const GraphWarpFieldRegularizationLayer& GraphWarpField::GetRegularizationLevel(int i_layer) const{
	return this->regularization_layers[i_layer];
}

int GraphWarpField::GetRegularizationLevelCount() const {
	return static_cast<int>(this->regularization_layers.size());
}

void GraphWarpField::BuildRegularizationLayers(int count, float decimation_radius) {
	this->regularization_layers = {};
	this->regularization_layers.emplace_back(GraphWarpFieldRegularizationLayer{decimation_radius, this->nodes});
	// this->regularization_layers.emplace_back(GraphWarpFieldRegularizationLayer{de});
	//TODO
}



} // namespace nnrt::geometry