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
	o3c::Tensor previous_layer_nodes = this->nodes;

	o3c::Tensor false_tensor(std::vector<bool>({false}), {1}, o3c::Bool, device);
	o3c::Tensor negative_one_tensor(std::vector<int32_t>({-1}), {1}, o3c::Int32, device);
	// build up node indices for each layer
	for (int i_layer = 1; i_layer < count; i_layer++) {
		auto& previous_layer = this->regularization_layers[i_layer - 1];

		// === find decimation "radius" and
		float current_decimation_radius = this->compute_layer_decimation_radius(i_layer, node_coverage);
		o3c::Tensor current_layer_node_index_sample, previous_layer_unfiltered_bin_node_indices;

		// median-grid-subsample the previous layer to find the indices of the previous layer nodes to use for the current layer, and the rest.
		std::tie(current_layer_node_index_sample, previous_layer_unfiltered_bin_node_indices) =
				geometry::functional::MedianGridSubsample3dPointsWithBinInfo(previous_layer_nodes, current_decimation_radius * 2,
				                                                             open3d::core::Int32);
		o3c::TensorKey current_layer_node_index_key = o3c::TensorKey::IndexTensor(current_layer_node_index_sample);

		// Separate-out the current layer nodes from the previous layer nodes to avoid duplicates to the previous layer. We can do this in multiple
		// ways, but a boolean mask seems to be the easiest to read and most efficient. Compute the mask first, based on the current layer sample.
		o3c::Tensor previous_layer_node_mask = o3c::Tensor({previous_layer_nodes.GetLength()}, o3c::Bool, device);
		previous_layer_node_mask.Fill(true);
		previous_layer_node_mask.SetItem(current_layer_node_index_key, false_tensor);
		o3c::TensorKey previous_layer_filter_key = o3c::TensorKey::IndexTensor(previous_layer_node_mask);

		o3c::Tensor previous_layer_filtered_node_indices = previous_layer.node_indices.GetItem(previous_layer_filter_key);

		// We want to set edge target indices to the indices of the actual virtual nodes; we already know that the indices of these
		// will go in increasing order and correspond to the span [running_node_count, running_node_count + updated_previous_layer_node_count).
		// we need to generate a mapping between the filtered previous layer local indices (i.e. 0, 1, 2, 3, 4, ..., 32 ) and the
		// embedding of the size of the unfiltered ones (i.e. 0, 0, 1, 0, 0, 2, 3, 0, ...., 30, when 1, 3, 4, )
		int64_t updated_previous_layer_node_count = previous_layer_filtered_node_indices.GetShape(0);
		o3c::Tensor previous_layer_filtered_virtual_node_indices = o3c::Tensor::Arange(0, updated_previous_layer_node_count, 1, o3c::Int32, device);
		o3c::Tensor previous_layer_unfiltered_embedding(previous_layer.node_indices.GetShape(), o3c::Int32, device);
		previous_layer_unfiltered_embedding.SetItem(previous_layer_filter_key, previous_layer_filtered_virtual_node_indices);

		// We have to retrieve the current layer indices before we proceed with replacing them with the filtered ones of the previous (finer) layer
		o3c::Tensor current_layer_indices = previous_layer.node_indices.GetItem(current_layer_node_index_key);
		o3c::Tensor current_layer_nodes = previous_layer_nodes.GetItem(current_layer_node_index_key);
		// Compile edge information based on which previous-layer nodes landed into the same bins as the sampled median ones.
		// This ensures that when we do coarse-to-fine ordering of the edges when laying out the Jacobian matrix for regularization residuals,
		// we get non-zero values exactly along the diagonal.
		o3c::Tensor layer_edges;
		kernel::warp_field::ReIndexLayerEdgeAdjacencyArray(layer_edges, max_vertex_degree, previous_layer_unfiltered_bin_node_indices,
		                                                   previous_layer_unfiltered_embedding);
		layer_edges = core::functional::SortTensorAlongLastDimension(layer_edges, true);

		// Safely update the previous layer with the filtered data now
		previous_layer.node_indices = previous_layer_filtered_node_indices;

		auto& current_layer = this->regularization_layers[i_layer];
		current_layer.decimation_radius = current_decimation_radius;
		current_layer.node_indices = current_layer_indices;
		current_layer.edges = layer_edges;

		previous_layer_nodes = current_layer_nodes;
	}
	int32_t virtual_node_count = 0;
	std::vector<int32_t> layer_node_count_inclusive_prefix_sum_data = {};
	for (int32_t i_layer = 0; i_layer < count - 1; i_layer++) {
		const auto& source_layer = this->regularization_layers[i_layer];
		virtual_node_count += static_cast<int32_t>(source_layer.node_indices.GetShape(0));
		layer_node_count_inclusive_prefix_sum_data.push_back(virtual_node_count);
	}

	// reorder layer nodes in edge order, proceed in reverse layer order
	for (int i_layer = count - 1; i_layer >= 1; i_layer--) {
		int32_t start_virtual_node_index = layer_node_count_inclusive_prefix_sum_data[i_layer - 1];
		auto& current_layer = this->regularization_layers[i_layer];
		auto& finer_layer = this->regularization_layers[i_layer - 1];
		o3c::Tensor finer_layer_local_edges_flattened = finest_layer.edges.Flatten();
		// // filter out -1 values
		o3c::Tensor negative_ones(finer_layer_local_edges_flattened.GetShape(), o3c::Int32, device);
		negative_ones.Fill(-1);
		o3c::TensorKey previous_layer_edge_mask = o3c::TensorKey::IndexTensor(
				finer_layer_local_edges_flattened.IsClose(negative_ones).LogicalNot());
		o3c::Tensor ordered_finer_layer_local_indices =
				finer_layer_local_edges_flattened.GetItem(previous_layer_edge_mask).To(o3c::Int64);
		o3c::TensorKey finer_layer_index = o3c::TensorKey::IndexTensor(ordered_finer_layer_local_indices);
		finer_layer.node_indices = finer_layer.node_indices.GetItem(finer_layer_index);
		if(i_layer > 1){
			finer_layer.edges = finer_layer.node_indices.GetItem(finer_layer_index);
		}
		int64_t edge_count = ordered_finer_layer_local_indices.GetLength();
		finer_layer_local_edges_flattened.SetItem(
				previous_layer_edge_mask,
				o3c::Tensor::Arange(start_virtual_node_index, start_virtual_node_index + edge_count, 1, o3c::Int32, device)
		);
		current_layer.edges = finer_layer_local_edges_flattened.Reshape({-1, max_vertex_degree});
		current_layer.edge_count = edge_count;
	}

	// convert array of structs to arrays of tensors, compute counts, layer weights, prefix sum
	std::vector<o3c::Tensor> layer_edge_sets_coarse_to_fine = {};
	std::vector<o3c::Tensor> layer_edge_layer_indices_coarse_to_fine = {};
	std::vector<o3c::Tensor> layer_node_index_sets_coarse_to_fine = {};
	std::vector<float> layer_decimation_radius_data = {};


	for (int32_t i_layer = 0; i_layer < count - 1; i_layer++) {
		const auto& source_layer = this->regularization_layers[i_layer];
		if (i_layer > 0) {
			layer_edge_sets_coarse_to_fine.push_back(source_layer.edges);
			const auto& target_layer = this->regularization_layers[i_layer + 1];
			layer_decimation_radius_data.push_back(target_layer.decimation_radius);
			o3c::Tensor layer_edge_layer_indices({source_layer.edges.GetShape(0)}, o3c::Int8, device);
			layer_edge_layer_indices.Fill(i_layer);
			layer_edge_layer_indices_coarse_to_fine.push_back(layer_edge_layer_indices);
		}
		layer_node_index_sets_coarse_to_fine.push_back(source_layer.node_indices);
	}

	this->virtual_node_index = o3c::Concatenate(layer_node_index_sets_coarse_to_fine);
	this->edges = o3c::Concatenate(layer_edge_sets_coarse_to_fine);
	this->edge_layer_indices = o3c::Concatenate(layer_edge_layer_indices_coarse_to_fine);

	o3c::Tensor layer_node_count_inclusive_prefix_sum(
			layer_node_count_inclusive_prefix_sum_data,
			{
					static_cast<int64_t>(layer_node_count_inclusive_prefix_sum_data.size())
			},
			o3c::Int32, device
	);

	o3c::Tensor layer_edge_weights =
			o3c::Tensor(layer_decimation_radius_data, {static_cast<int64_t>(layer_decimation_radius_data.size())}, o3c::Float32, device);
}

const o3c::Tensor& HierarchicalGraphWarpField::GetEdges() const {
	return this->edges;
}


const o3c::Tensor& HierarchicalGraphWarpField::GetVirtualNodeIndices() const {
	return this->virtual_node_index;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetNodes(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_nodes.Get(&this->virtual_node_index) : this->nodes;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetTranslations(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_translations.Get(&this->virtual_node_index) : this->nodes;
}

const o3c::Tensor& HierarchicalGraphWarpField::GetRotations(bool use_virtual_ordering) {
	return use_virtual_ordering ? this->indexed_translations.Get(&this->virtual_node_index) : this->nodes;
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