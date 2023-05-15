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
#include <algorithm>
// 3rd party
#include <open3d/core/TensorFunction.h>
// local
#include "geometry/GraphWarpField.h"
#include "core/linalg/Matmul3D.h"
#include "geometry/functional/WarpAnchorComputation.h"
#include "geometry/functional/GeometrySampling.h"
#include "geometry/kernel/HierarchicalGraphWarpField.h"
#include "core/functional/Sorting.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::geometry {


WarpField::WarpField(
		o3c::Tensor nodes,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count
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
	o3c::AssertTensorShape(this->nodes, { node_count, 3 });
	o3c::AssertTensorDtype(this->nodes, o3c::Float32);

	this->ResetRotations();
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

open3d::core::Device WarpField::GetDevice() const {
	return this->nodes.GetDevice();
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
	auto device = this->nodes.GetDevice();
	int64_t node_count = this->nodes.GetLength();
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

HierarchicalGraphWarpField::HierarchicalGraphWarpField(
		open3d::core::Tensor nodes,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count,
		int layer_count,
		int max_vertex_degree,
		std::function<float(int, float)> compute_layer_decimation_radius
) : WarpField(std::move(nodes), node_coverage, threshold_nodes_by_distance_by_default, anchor_count, minimum_valid_anchor_count),
    regularization_layers(), compute_layer_decimation_radius(std::move(compute_layer_decimation_radius)) {
	this->RebuildRegularizationLayers(layer_count, max_vertex_degree);
}


const RegularizationLayer& HierarchicalGraphWarpField::GetRegularizationLevel(int i_layer) const {
	return this->regularization_layers[i_layer];
}

int HierarchicalGraphWarpField::GetRegularizationLevelCount() const {
	return static_cast<int>(this->regularization_layers.size());
}

void HierarchicalGraphWarpField::RebuildRegularizationLayers(int count, int max_vertex_degree) {
	this->regularization_layers.resize(count);
	auto& finest_layer = this->regularization_layers[0];
	// use all nodes at first for finest layer
	finest_layer.node_coverage = this->node_coverage;
	o3c::Device device = this->GetDevice();
	finest_layer.node_indices = o3c::Tensor::Arange(0, this->nodes.GetLength(), 1, o3c::Int32, device);
	o3c::Tensor previous_layer_nodes = this->nodes;

	o3c::Tensor false_tensor(std::vector<bool>({false}), {1}, o3c::Bool, device);

	// build up node indices for each layer
	for (int i_layer = 1; i_layer < count; i_layer++) {
		auto& previous_layer = this->regularization_layers[i_layer - 1];

		// === find decimation "radius" and
		float current_decimation_radius = this->compute_layer_decimation_radius(i_layer, node_coverage);

		// median-grid-subsample the previous layer to find the indices of the previous layer nodes to use for the current layer, and the rest.
		auto [current_layer_node_index_sample, previous_layer_unfiltered_bin_node_indices] =
				geometry::functional::MedianGridSubsample3dPointsWithBinInfo(previous_layer_nodes, current_decimation_radius * 2);

		o3c::TensorKey current_layer_node_index_key = o3c::TensorKey::IndexTensor(current_layer_node_index_sample);

		// Separate-out the current layer nodes from the previous layer nodes to avoid duplicates to the previous layer. We can do this in multiple
		// ways, but a boolean mask seems to be the easiest to read and most efficient. Compute the mask first, based on the current layer sample.
		o3c::Tensor previous_layer_node_mask = o3c::Tensor({previous_layer_nodes.GetLength()}, o3c::Bool, device);
		previous_layer_node_mask.Fill(true);
		previous_layer_node_mask.SetItem(current_layer_node_index_key, false_tensor);
		// we have to retrieve the current layer indices before we proceed with the filtering of the previous (finer) layer
		o3c::Tensor current_layer_indices = previous_layer.node_indices.GetItem(current_layer_node_index_key);
		o3c::Tensor current_layer_nodes = previous_layer_nodes.GetItem(current_layer_node_index_key);
		// Compile edge information based on which previous-layer nodes landed into the same bins as the sampled median ones.
		// This ensures that when we do coarse-to-fine ordering of the edges when laying out the Jacobian matrix, we get non-zero
		// values exactly along the diagonal.
		o3c::Tensor layer_edges;
		kernel::warp_field::PrepareLayerEdges(layer_edges, previous_layer_unfiltered_bin_node_indices, previous_layer.node_indices);

		// Safely apply the previous layer filtering now
		o3c::TensorKey previous_layer_filter_key = o3c::TensorKey::IndexTensor(previous_layer_node_mask);
		previous_layer.node_indices = previous_layer.node_indices.GetItem(previous_layer_filter_key);
		previous_layer_nodes = previous_layer_nodes.GetItem(previous_layer_filter_key);

		auto& current_layer = this->regularization_layers[i_layer];
		current_layer.node_coverage = current_decimation_radius;
		current_layer.node_indices = current_layer_indices;
		current_layer.edges = layer_edges;

		previous_layer_nodes = current_layer_nodes;
	}

	// convert array of structs to arrays of tensors, compute counts, layer weights, prefix sum
	std::vector<o3c::Tensor> layer_edge_sets;
	std::vector<o3c::Tensor> node_index_sets;
	std::vector<float> layer_edge_weight_data = {};
	int32_t virtual_node_count = 0;
	std::vector<int32_t> layer_node_count_inclusive_prefix_sum_data = {};

	for (int i_layer = 0; i_layer < count; i_layer++) {
		const auto& source_layer = this->regularization_layers[i_layer];
		virtual_node_count += static_cast<int32_t>(source_layer.node_indices.GetLength());
		layer_node_count_inclusive_prefix_sum_data.push_back(virtual_node_count);

		if (i_layer < count - 1) {
			layer_edge_sets.push_back(source_layer.edges);
			const auto& target_layer = this->regularization_layers[i_layer + 1];
			layer_edge_weight_data.push_back(target_layer.node_coverage);
		}
		node_index_sets.push_back(source_layer.node_indices);
	}

	// concatenate array data, tensor-ize
	o3c::Tensor concatenated_edges = o3c::Concatenate(layer_edge_sets, 0);
	this->virtual_node_indices = o3c::Concatenate(node_index_sets);
	o3c::Tensor layer_edge_weights = o3c::Tensor(layer_edge_weight_data, {static_cast<int64_t>(layer_edge_weight_data.size())}, o3c::Float32, device);
	o3c::Tensor layer_node_count_inclusive_prefix_sum(
			layer_node_count_inclusive_prefix_sum_data,
			{static_cast<int64_t>(layer_node_count_inclusive_prefix_sum_data.size())},
			o3c::Int32, device
	);

	kernel::warp_field::FlattenWarpField(this->edges, this->edge_weights, concatenated_edges, this->virtual_node_indices, layer_edge_weights,
	                                     layer_node_count_inclusive_prefix_sum);
}

const o3c::Tensor& HierarchicalGraphWarpField::GetEdges() const {
	return this->edges;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetEdgeWeights() const {
	return this->edge_weights;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetVirtualNodeIndices() const {
	return this->virtual_node_indices;
}

} // namespace nnrt::geometry