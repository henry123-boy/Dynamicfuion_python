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
// stdlib includes

// third-party includes
#include <open3d/utility/Logging.h>

// local includes
#include "geometry/kernel/HierarchicalGraphWarpField.h"
#include "core/DeviceSelection.h"


namespace utility = open3d::utility;

namespace nnrt::geometry::kernel::warp_field {

void
FlattenWarpField(
		open3d::core::Tensor& edges,
		open3d::core::Tensor& edge_weights,
		const open3d::core::Tensor& concatenated_layer_edges,
		const open3d::core::Tensor& concatenated_layer_node_indices,
		const open3d::core::Tensor& layer_edge_weights,
		const open3d::core::Tensor& layer_virtual_node_count_prefix_sum
) {

	core::ExecuteOnDevice(
			concatenated_layer_edges.GetDevice(),
			[&] {
				FlattenWarpField<open3d::core::Device::DeviceType::CPU>(edges, edge_weights, concatenated_layer_edges,
				                                                        concatenated_layer_node_indices, layer_edge_weights,
				                                                        layer_virtual_node_count_prefix_sum);
			},
			[&] {
				NNRT_IF_CUDA(FlattenWarpField<open3d::core::Device::DeviceType::CUDA>(edges, edge_weights, concatenated_layer_edges,
				                                                                      concatenated_layer_node_indices, layer_edge_weights,
				                                                                      layer_virtual_node_count_prefix_sum););
			}
	);
}

void ReIndexLayerEdgeAdjacencyArray(
		open3d::core::Tensor& edges,
		int32_t max_vertex_degree,
		const open3d::core::Tensor& previous_layer_unfiltered_local_bin_node_indices,
		const open3d::core::Tensor& previous_layer_unfiltered_global_node_indices
) {
	core::ExecuteOnDevice(
			previous_layer_unfiltered_local_bin_node_indices.GetDevice(),
			[&] {
				ReIndexLayerEdgeAdjacencyArray<open3d::core::Device::DeviceType::CPU>(edges, max_vertex_degree,
				                                                                      previous_layer_unfiltered_local_bin_node_indices,
				                                                                      previous_layer_unfiltered_global_node_indices);
			},
			[&] {
				NNRT_IF_CUDA(ReIndexLayerEdgeAdjacencyArray<open3d::core::Device::DeviceType::CUDA>(edges, max_vertex_degree,
				                                                                                    previous_layer_unfiltered_local_bin_node_indices,
				                                                                                    previous_layer_unfiltered_global_node_indices););
			}
	);
}

void AdjacencyArrayToEdgesWithDuplicateTargetFilteredOut(
		open3d::core::Tensor& edges,
		const open3d::core::Tensor& adjacency_array,
		const open3d::core::Tensor& source_node_indices,
		const open3d::core::Tensor& target_node_indices
) {
	core::ExecuteOnDevice(
			adjacency_array.GetDevice(),
			[&] {
				AdjacencyArrayToEdgesWithDuplicateTargetFilteredOut<open3d::core::Device::DeviceType::CPU>(
						edges,
						adjacency_array,
						source_node_indices,
						target_node_indices
				);
			},
			[&] {
				NNRT_IF_CUDA(
						AdjacencyArrayToEdgesWithDuplicateTargetFilteredOut<open3d::core::Device::DeviceType::CUDA>(
								edges,
								adjacency_array,
								source_node_indices,
								target_node_indices
						);
				);
			}
	);
}


} // namespace nnrt::geometry::kernel::warp_field