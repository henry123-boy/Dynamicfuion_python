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
#include "geometry/HierarchicalGraphWarpField.h"
#include "geometry/functional/GeometrySampling.h"
#include "geometry/kernel/HierarchicalGraphWarpField.h"


#include "core/functional/Sorting.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::geometry {
HierarchicalGraphWarpField::HierarchicalGraphWarpField(
		open3d::core::Tensor nodes,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count,
		int layer_count,
		int max_vertex_degree,
		std::function<float(int, float)> compute_layer_decimation_radius
) : WarpField(std::move(nodes),
              node_coverage,
              threshold_nodes_by_distance_by_default,
              anchor_count,
              minimum_valid_anchor_count),
    regularization_layers(),
    compute_layer_decimation_radius(std::move(compute_layer_decimation_radius)),
    indexed_nodes(this->nodes),
    indexed_rotations(this->rotations),
    indexed_translations(this->translations) {
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
	finest_layer.decimation_radius = this->node_coverage;
	o3c::Device device = this->GetDevice();
	finest_layer.node_indices = o3c::Tensor::Arange(0, this->nodes.GetLength(), 1, o3c::Int32, device);
	finest_layer.node_positions = this->nodes;

	o3c::Tensor false_tensor(std::vector<bool>({false}), {1}, o3c::Bool, device);
	// build up node indices for each coarser layer, while filtering them out of the finer layer
	for (int i_layer = 1; i_layer < count; i_layer++) {
		auto& finer_layer = this->regularization_layers[i_layer - 1];
		auto& current_layer = this->regularization_layers[i_layer];

		// === find decimation "radius" and
		current_layer.decimation_radius = this->compute_layer_decimation_radius(i_layer, node_coverage);

		// median-grid-subsample the previous layer to find the indices of the previous layer nodes to use for the current layer
		o3c::Tensor current_layer_node_index_sample =
				geometry::functional::MedianGridSubsample3dPoints(finer_layer.node_positions, current_layer.decimation_radius * 2);

		o3c::TensorKey current_layer_node_index_key = o3c::TensorKey::IndexTensor(current_layer_node_index_sample);

		// Separate-out the current layer nodes from the previous layer nodes to avoid duplicates to the previous layer.
		// Compute the mask first, based on the current layer sample.
		o3c::Tensor finer_layer_node_mask = o3c::Tensor({finer_layer.node_indices.GetLength()}, o3c::Bool, device);
		finer_layer_node_mask.Fill(true);
		finer_layer_node_mask.SetItem(current_layer_node_index_key, false_tensor);
		o3c::TensorKey previous_layer_filter_key = o3c::TensorKey::IndexTensor(finer_layer_node_mask);

		// We have to retrieve the current layer's node indices from finer layer's indices before we proceed with filtering the finter layer.
		current_layer.node_indices = finer_layer.node_indices.GetItem(current_layer_node_index_key);
		current_layer.node_positions = finer_layer.node_positions.GetItem(current_layer_node_index_key);

		// Safely filter the finer layer's nodes now
		finer_layer.node_indices = finer_layer.node_indices.GetItem(previous_layer_filter_key);
		finer_layer.node_positions = finer_layer.node_positions.GetItem(previous_layer_filter_key);
	}
	// count up the nodes in each layer
	int32_t virtual_node_count = 0;
	std::vector<int32_t> layer_node_count_exclusive_prefix_sum_data = {};
	std::vector<o3c::Tensor> layer_node_index_sets_fine_to_coarse = {};
	for (int32_t i_layer = 0; i_layer < count; i_layer++) {
		const auto& layer = this->regularization_layers[i_layer];
		layer_node_count_exclusive_prefix_sum_data.push_back(virtual_node_count);
		virtual_node_count += static_cast<int32_t>(layer.node_indices.GetShape(0));
		layer_node_index_sets_fine_to_coarse.push_back(layer.node_indices);
	}
	// Now that we have precise node sets ready, in the next pass, connect each successive pair of layers with edges.
	// Since we'll need to reshuffle the finer layer indices (and edges) as we proceed, we'll need to do this in top-down, coarsest-to-finest layer
	// order.

	for (int32_t i_layer = count - 1; i_layer >= 1; i_layer--) {
		int32_t finer_layer_first_virtual_node_index = layer_node_count_exclusive_prefix_sum_data[i_layer - 1];
		int32_t current_layer_first_virtual_node_index = layer_node_count_exclusive_prefix_sum_data[i_layer];
		auto& current_layer = this->regularization_layers[i_layer];
		auto& finer_layer = this->regularization_layers[i_layer - 1];

		// find K nearest neighbors in the finer level for each source node
		o3c::Tensor finer_layer_adjacency_array, squared_distances;
		core::KdTree current_layer_node_tree(current_layer.node_positions);
		current_layer_node_tree.FindKNearestToPoints(finer_layer_adjacency_array, squared_distances,
		                                             finer_layer.node_positions, max_vertex_degree, false);
		finer_layer_adjacency_array = core::functional::SortTensorAlongLastDimension(finer_layer_adjacency_array, true, core::functional::SortOrder::DESC);

		// our goal now is to reorder nodes in the finer layer such that:
		//  1) consecutive edges are laid out as if grouped by their source node index, in descending source node index order ("coarse-to-fine")
		//  2) the targets of consecutive edges have descending (possibly, non-consecutive) indices and start from the last virtual node index in layer
		// in our edge definitions, we need to use virtual node indices for both source and target nodes
		o3c::Tensor finer_layer_edges;
		o3c::Tensor finter_layer_virtual_node_indices =
				o3c::Tensor::Arange(finer_layer_first_virtual_node_index, finer_layer_first_virtual_node_index + finer_layer.node_indices.GetLength(),
				                    1, o3c::Int32, device);
		o3c::Tensor current_layer_virtual_node_indices =
				o3c::Tensor::Arange(current_layer_first_virtual_node_index,
				                    current_layer_first_virtual_node_index + current_layer.node_indices.GetLength(),
				                    1, o3c::Int32, device);
		kernel::warp_field::AdjacencyArrayToEdges(
				finer_layer_edges, finer_layer_adjacency_array, finter_layer_virtual_node_indices, current_layer_virtual_node_indices, true);

		finer_layer.edges = finer_layer_edges;
	}

	// convert array of structs to arrays of tensors, compute counts, edge layer indices, decimation radii
	std::vector<o3c::Tensor> layer_edge_sets_coarse_to_fine = {};
	std::vector<o3c::Tensor> layer_edge_layer_indices_coarse_to_fine = {};

	std::vector<float> layer_decimation_radius_data = {};

	for (int32_t i_layer = count - 1; i_layer >= 0; i_layer--) {
		const auto& source_layer = this->regularization_layers[i_layer];
		if (i_layer < count - 1) {
			layer_edge_sets_coarse_to_fine.push_back(source_layer.edges);
			const auto& target_layer = this->regularization_layers[i_layer + 1];
			layer_decimation_radius_data.push_back(target_layer.decimation_radius);
			o3c::Tensor layer_edge_layer_indices({source_layer.edges.GetShape(0)}, o3c::Int8, device);
			layer_edge_layer_indices.Fill(i_layer);
			layer_edge_layer_indices_coarse_to_fine.push_back(layer_edge_layer_indices);
		}
	}

	this->node_indices = o3c::Concatenate(layer_node_index_sets_fine_to_coarse);
	this->edges = o3c::Concatenate(layer_edge_sets_coarse_to_fine);
	this->edge_layer_indices = o3c::Concatenate(layer_edge_layer_indices_coarse_to_fine);
	this->layer_decimation_radii =
			o3c::Tensor(layer_decimation_radius_data, {static_cast<int64_t>(layer_decimation_radius_data.size())}, o3c::Float32, device);
}

const o3c::Tensor& HierarchicalGraphWarpField::GetEdges() const {
	return this->edges;
}


const o3c::Tensor& HierarchicalGraphWarpField::GetVirtualNodeIndices() const {
	return this->node_indices;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetNodes(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_nodes.Get(&this->node_indices) : this->nodes;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetTranslations(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_translations.Get(&this->node_indices) : this->nodes;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetRotations(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_translations.Get(&this->node_indices) : this->nodes;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetEdgeLayerIndices() const {
	return this->edge_layer_indices;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetLayerDecimationRadii() const {
	return this->layer_decimation_radii;
}


HierarchicalGraphWarpField::ReindexedTensorWrapper::ReindexedTensorWrapper(const o3c::Tensor* index, const o3c::Tensor& source_tensor) :
		linear_index(index), source_tensor(source_tensor) {
	Reindex();
}

HierarchicalGraphWarpField::ReindexedTensorWrapper::ReindexedTensorWrapper(const o3c::Tensor& source_tensor) :
		linear_index(nullptr), source_tensor(source_tensor) {
}

void HierarchicalGraphWarpField::ReindexedTensorWrapper::Reindex() {
	this->reindexed_tensor = this->source_tensor.GetItem(o3c::TensorKey::IndexTensor(*this->linear_index));
}

const o3c::Tensor& HierarchicalGraphWarpField::ReindexedTensorWrapper::Get(const o3c::Tensor* index) {
	if (!this->linear_index->IsSame(*index)) {
		this->linear_index = index;
		Reindex();
	}
	return this->reindexed_tensor;
}


} // namespace nnrt::geometry