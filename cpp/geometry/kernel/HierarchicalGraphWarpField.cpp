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

void PrepareLayerEdges(
		open3d::core::Tensor& edges,
		const open3d::core::Tensor& previous_layer_unfiltered_local_bin_node_indices,
		const open3d::core::Tensor& previous_layer_unfiltered_global_node_indices
) {
	core::ExecuteOnDevice(
			previous_layer_unfiltered_local_bin_node_indices.GetDevice(),
			[&] {
				PrepareLayerEdges<open3d::core::Device::DeviceType::CPU>(edges, previous_layer_unfiltered_local_bin_node_indices,
				                                                         previous_layer_unfiltered_global_node_indices);
			},
			[&] {
				NNRT_IF_CUDA(PrepareLayerEdges<open3d::core::Device::DeviceType::CUDA>(edges, previous_layer_unfiltered_local_bin_node_indices,
				                                                                       previous_layer_unfiltered_global_node_indices););
			}
	);
}

} // namespace nnrt::geometry::kernel::warp_field