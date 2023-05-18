//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/11/23.
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
#include <open3d/core/ParallelFor.h>

// local includes
#include "geometry/kernel/HierarchicalGraphWarpField.h"
#include "core/platform_independence/Atomics.h"
#include "core/platform_independence/AtomicCounterArray.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::warp_field {

//TODO: not sure that this reindexing even needs to exist. Remove or finish.
template<open3d::core::Device::DeviceType TDeviceType>
void FlattenWarpField(
		open3d::core::Tensor& edges,
		open3d::core::Tensor& edge_weights,
		const open3d::core::Tensor& concatenated_layer_edges,
		const open3d::core::Tensor& concatenated_layer_node_indices,
		const open3d::core::Tensor& layer_edge_weights,
		const open3d::core::Tensor& layer_virtual_node_count_inclusive_prefix_sum
) {
	// === get counts, check dimensions
	o3c::Device device = concatenated_layer_edges.GetDevice();
	int64_t virtual_source_node_count = concatenated_layer_edges.GetLength();
	int64_t virtual_node_count = concatenated_layer_node_indices.GetLength();
	int64_t max_vertex_degree = concatenated_layer_edges.GetShape(1);

	o3c::AssertTensorShape(concatenated_layer_edges, { virtual_source_node_count, max_vertex_degree });
	o3c::AssertTensorDtype(concatenated_layer_edges, o3c::Int32);
	o3c::AssertTensorDevice(concatenated_layer_edges, device);

	o3c::AssertTensorShape(concatenated_layer_node_indices, { virtual_node_count });
	o3c::AssertTensorDtype(concatenated_layer_node_indices, o3c::Int32);
	o3c::AssertTensorDevice(concatenated_layer_node_indices, device);

	int64_t source_layer_count = layer_edge_weights.GetLength();
	o3c::AssertTensorShape(layer_edge_weights, { source_layer_count });
	o3c::AssertTensorDtype(layer_edge_weights, o3c::Float32);
	o3c::AssertTensorDevice(layer_edge_weights, device);

	o3c::AssertTensorShape(layer_virtual_node_count_inclusive_prefix_sum, { source_layer_count + 1 });
	o3c::AssertTensorDtype(layer_virtual_node_count_inclusive_prefix_sum, o3c::Int32);
	o3c::AssertTensorDevice(layer_virtual_node_count_inclusive_prefix_sum, device);

	// === initialize output data structures
	edges = o3c::Tensor({virtual_source_node_count, max_vertex_degree}, o3c::Int32, device);
	auto edge_data = edges.GetDataPtr<int32_t>();
	edge_weights = o3c::Tensor({virtual_source_node_count}, o3c::Float32, device);
	auto edge_weight_data = edge_weights.GetDataPtr<float>();

	// === get pointers to input data
	const auto layer_edge_data = concatenated_layer_edges.GetDataPtr<int32_t>();
	const auto layer_node_index_data = concatenated_layer_node_indices.GetDataPtr<int32_t>();
	const auto layer_edge_weight_data = layer_edge_weights.GetDataPtr<float>();
	const auto layer_cumulative_node_counts = layer_virtual_node_count_inclusive_prefix_sum.GetDataPtr<int32_t>();

	o3c::ParallelFor(
			device,
			virtual_source_node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_source_virtual_vertex) {
				// find layer index
				int32_t i_layer = 0;
				for (; i_layer < source_layer_count && i_source_virtual_vertex >= layer_cumulative_node_counts[i_layer]; i_layer++);
				const auto* next_layer_node_indices = layer_node_index_data + layer_cumulative_node_counts[i_layer];
				float edge_weight = layer_edge_weight_data[i_layer];
				edge_weight_data[i_source_virtual_vertex] = edge_weight;
				auto edges_start_index = i_source_virtual_vertex * max_vertex_degree;
				const auto source_vertex_layer_edges = layer_edge_data + edges_start_index;
				// relies on all-virtual-vertex layer_node_index to always contain source vertices first, i.e. fine-to-coarse layer ordering
				int source_vertex_index = layer_node_index_data[i_source_virtual_vertex];

				for (int i_vertex_edge = 0; i_vertex_edge < max_vertex_degree; i_vertex_edge++) {
					int32_t i_target_index_in_target_layer = source_vertex_layer_edges[i_vertex_edge];
					int64_t edge_index = edges_start_index + i_vertex_edge;
					if (i_target_index_in_target_layer == -1 || next_layer_node_indices[i_target_index_in_target_layer] == source_vertex_index) {
						edge_data[edge_index] = -1;
						// assume the source_vertex_layer_edges values are sorted, so sentinel value -1 always appears in the end.
						continue;
					} else {
						int32_t target_node_index = next_layer_node_indices[i_target_index_in_target_layer];
						edge_data[edge_index] = target_node_index;
					}
				}

			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void ReIndexLayerEdgeAdjacencyArray(
		open3d::core::Tensor& edges,
		int32_t max_vertex_degree,
		const open3d::core::Tensor& previous_layer_unfiltered_local_bin_node_indices,
		const open3d::core::Tensor& previous_layer_unfiltered_virtual_node_indices
) {
	// counters and checks
	o3c::Device device = previous_layer_unfiltered_local_bin_node_indices.GetDevice();

	int64_t current_layer_node_count = previous_layer_unfiltered_local_bin_node_indices.GetShape(0);
	int64_t max_vertex_degree_based_on_data = previous_layer_unfiltered_local_bin_node_indices.GetShape(1);
	if (max_vertex_degree < max_vertex_degree_based_on_data) {
		open3d::utility::LogError("max_vertex_degree specified ({}) is lower than the allowed maximum, (largest bin count-1), "
		                          "which is {} based on passed-in data. Try increasing max_vertex_degree.", max_vertex_degree,
		                          max_vertex_degree_based_on_data);
	}
	o3c::AssertTensorShape(previous_layer_unfiltered_local_bin_node_indices, { current_layer_node_count, max_vertex_degree_based_on_data });
	o3c::AssertTensorDtype(previous_layer_unfiltered_local_bin_node_indices, o3c::Int32);

	int64_t previous_layer_unfiltered_node_count = previous_layer_unfiltered_virtual_node_indices.GetLength();
	o3c::AssertTensorShape(previous_layer_unfiltered_virtual_node_indices, { previous_layer_unfiltered_node_count });
	o3c::AssertTensorDtype(previous_layer_unfiltered_virtual_node_indices, o3c::Int32);
	o3c::AssertTensorDevice(previous_layer_unfiltered_virtual_node_indices, device);

	// prepare inputs
	auto target_local_index_data = previous_layer_unfiltered_local_bin_node_indices.GetDataPtr<int32_t>();
	auto target_global_index_data = previous_layer_unfiltered_virtual_node_indices.GetDataPtr<int32_t>();

	// prepare output
	edges = o3c::Tensor({current_layer_node_count, max_vertex_degree}, o3c::Int32, device);
	edges.Fill(-1);

	auto edge_data = edges.GetDataPtr<int32_t>();
	o3c::ParallelFor(
			device, current_layer_node_count * max_vertex_degree_based_on_data,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t global_edge_index) {
				int64_t i_source_vertex_in_layer = global_edge_index / max_vertex_degree_based_on_data;
				int64_t i_target_vertex_in_layer = global_edge_index % max_vertex_degree_based_on_data;
				auto local_target_index = target_local_index_data[global_edge_index];
				if (local_target_index != -1) {
					auto global_target_index = target_global_index_data[local_target_index];
					edge_data[i_source_vertex_in_layer * max_vertex_degree + i_target_vertex_in_layer] = global_target_index;
				}
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void AdjacencyArrayToEdgesWithDuplicateTargetFilteredOut(
		open3d::core::Tensor& edges,
		const open3d::core::Tensor& adjacency_array,
		const open3d::core::Tensor& source_node_indices,
		const open3d::core::Tensor& target_node_indices
) {
	// counters & checks
	o3c::Device device = adjacency_array.GetDevice();
	int64_t source_node_count = adjacency_array.GetShape(0);
	int64_t max_vertex_degree = adjacency_array.GetShape(1);
	int64_t target_node_count = target_node_indices.GetShape(0);
	o3c::AssertTensorShape(adjacency_array, { source_node_count, max_vertex_degree });
	o3c::AssertTensorDtype(adjacency_array, o3c::Int32);

	o3c::AssertTensorDevice(source_node_indices, device);
	o3c::AssertTensorShape(source_node_indices, { source_node_count });
	o3c::AssertTensorDtype(source_node_indices, o3c::Int32);

	o3c::AssertTensorDevice(target_node_indices, device);
	o3c::AssertTensorShape(target_node_indices, { target_node_count });
	o3c::AssertTensorDtype(target_node_indices, o3c::Int32);

	// prepare input
	auto adjacency_data = adjacency_array.GetDataPtr<int32_t>();
	auto source_index_data = source_node_indices.GetDataPtr<int32_t>();
	auto target_index_data = target_node_indices.GetDataPtr<int32_t>();

	// === prepare output & locks
	// If we just make edges of size {target_node_count, 2} and then fill in according to the target index,
	// we mess up the ordering of the source nodes in the edge array.
	// To prevent this, we first make an array of size {source_node_count * max_vertex_degree, 2}, and later
	// filter out the "empty" edges. This way, we can preserve edge ordering by source node in a parallel way.
	edges = o3c::Tensor({source_node_count * max_vertex_degree, 2}, o3c::Int32);
	auto edge_data = edges.GetDataPtr<int32_t>();
	core::AtomicCounterArray<TDeviceType> target_fence(target_node_count);

	o3c::ParallelFor(
			device, source_node_count * max_vertex_degree,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t global_adjacency_index) {
				int64_t i_source = global_adjacency_index / max_vertex_degree;
				int64_t i_source_adjacency = global_adjacency_index % max_vertex_degree;

				int32_t i_target = adjacency_data[max_vertex_degree * i_source + i_source_adjacency];
				auto edge_out = edge_data + (i_source * max_vertex_degree + i_source_adjacency) * 2;
				if (i_target != -1 && target_fence.FetchAdd(i_target, 1) == 0) {
					edge_out[0] = source_index_data[i_source];
					edge_out[1] = target_index_data[i_target];
				} else {
					edge_out[0] = -1;
					edge_out[1] = -1;
				}
			}
	);

	// now we filter out the "empty" edges
	o3c::Tensor negative_one_tensor(std::vector<int32_t>({-1, -1}), {2}, o3c::Int32, device);
	edges = edges.GetItem(o3c::TensorKey::IndexTensor(edges != negative_one_tensor));
}


template<open3d::core::Device::DeviceType TDeviceType>
void AdjacencyArrayToEdges(
		open3d::core::Tensor& edges,
		const open3d::core::Tensor& adjacency_array,
		const open3d::core::Tensor& source_node_indices,
		const open3d::core::Tensor& target_node_indices,
		bool flip_source_order
) {
	// counters & checks
	o3c::Device device = adjacency_array.GetDevice();
	int64_t source_node_count = adjacency_array.GetShape(0);
	int64_t max_vertex_degree = adjacency_array.GetShape(1);
	int64_t target_node_count = target_node_indices.GetShape(0);
	o3c::AssertTensorShape(adjacency_array, { source_node_count, max_vertex_degree });
	o3c::AssertTensorDtype(adjacency_array, o3c::Int32);

	o3c::AssertTensorDevice(source_node_indices, device);
	o3c::AssertTensorShape(source_node_indices, { source_node_count });
	o3c::AssertTensorDtype(source_node_indices, o3c::Int32);

	o3c::AssertTensorDevice(target_node_indices, device);
	o3c::AssertTensorShape(target_node_indices, { target_node_count });
	o3c::AssertTensorDtype(target_node_indices, o3c::Int32);

	// prepare input
	auto adjacency_data = adjacency_array.GetDataPtr<int32_t>();
	auto source_index_data = source_node_indices.GetDataPtr<int32_t>();
	auto target_index_data = target_node_indices.GetDataPtr<int32_t>();

	// === prepare output
	// If we just make edges of size {target_node_count, 2} and then fill in according to the target index,
	// we mess up the ordering of the source nodes in the edge array and lose edges with different source but same target.
	// To prevent this, we first make an array of size {source_node_count * max_vertex_degree, 2}, and later
	// filter out the "empty" edges. This way, we can preserve edge ordering by source node in a parallel way.
	edges = o3c::Tensor({source_node_count * max_vertex_degree, 2}, o3c::Int32, device);
	auto edge_data = edges.GetDataPtr<int32_t>();

	if (flip_source_order) {
		o3c::ParallelFor(
				device, source_node_count * max_vertex_degree,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t global_adjacency_index) {
					int64_t i_source = global_adjacency_index / max_vertex_degree;
					int64_t i_source_adjacency = global_adjacency_index % max_vertex_degree;

					int32_t i_target = adjacency_data[max_vertex_degree * i_source + i_source_adjacency];
					auto edge_out = edge_data + ((source_node_count - 1 - i_source) * max_vertex_degree + i_source_adjacency) * 2;
					if (i_target != -1) {
						edge_out[0] = source_index_data[i_source];
						edge_out[1] = target_index_data[i_target];
					} else {
						edge_out[0] = -1;
						edge_out[1] = -1;
					}
				}
		);
	} else {
		o3c::ParallelFor(
				device, source_node_count * max_vertex_degree,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t global_adjacency_index) {
					int64_t i_source = global_adjacency_index / max_vertex_degree;
					int64_t i_source_adjacency = global_adjacency_index % max_vertex_degree;

					int32_t i_target = adjacency_data[max_vertex_degree * i_source + i_source_adjacency];
					auto edge_out = edge_data + (i_source * max_vertex_degree + i_source_adjacency) * 2;
					if (i_target != -1) {
						edge_out[0] = source_index_data[i_source];
						edge_out[1] = target_index_data[i_target];
					} else {
						edge_out[0] = -1;
						edge_out[1] = -1;
					}
				}
		);
	}


	// now we filter out the "empty" edges
	o3c::Tensor negative_one_tensor(std::vector<int32_t>({-1}), {1}, o3c::Int32, device);
	edges = edges.GetItem(o3c::TensorKey::IndexTensor(edges != negative_one_tensor)).Reshape({-1, 2});
}

} // namespace nnrt::geometry::kernel::warp_field