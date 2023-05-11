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

namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::warp_field {

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
	edge_weights = o3c::Tensor({virtual_source_node_count, max_vertex_degree}, o3c::Float32, device);
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
				const auto source_vertex_layer_edges = layer_edge_data + i_source_virtual_vertex;

				for (int i_vertex_edge = 0; i_vertex_edge < max_vertex_degree; i_vertex_edge++) {
					int32_t i_target_index_in_target_layer = source_vertex_layer_edges[i_vertex_edge];
					int64_t edge_index = i_source_virtual_vertex * max_vertex_degree + i_vertex_edge;
					if (i_target_index_in_target_layer == -1) {
						edge_data[edge_index] = -1;
						edge_weight_data[edge_index] = 0.0f;
						// assume the source_vertex_layer_edges values are sorted, so sentinel value -1 always appears in the end.
						continue;
					} else {
						int32_t target_node_index = next_layer_node_indices[i_target_index_in_target_layer];
						edge_data[edge_index] = target_node_index;
						edge_weight_data[edge_index] = edge_weight;
					}
				}
			}
	);
}

} // namespace nnrt::geometry::kernel::warp_field