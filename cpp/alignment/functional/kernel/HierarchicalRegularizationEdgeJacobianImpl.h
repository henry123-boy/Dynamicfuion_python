//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/20/23.
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
#include <open3d/core/Device.h>
#include <open3d/core/Tensor.h>
#include <open3d/core/ParallelFor.h>

// local includes
#include "alignment/functional/kernel/HierarchicalRegularizationEdgeJacobian.h"
#include "core/platform_independence/AtomicCounterArray.h"

namespace o3c = open3d::core;

namespace nnrt::alignment::functional::kernel {

#define MAX_EDGES_PER_NODE 100

template <open3d::core::Device::DeviceType TDeviceType>
void HierarchicalRegularizationEdgeJacobiansAndNodeAssociations(
		open3d::core::Tensor& edge_jacobians,
		open3d::core::Tensor& node_edge_jacobian_indices_jagged,
		open3d::core::Tensor& node_edge_jacobian_counts,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& edge_layer_indices,
		const open3d::core::Tensor& layer_decimation_radii
) {
	// === counters & checks
	o3c::Device device = node_positions.GetDevice();

	int64_t node_count = node_positions.GetLength();
	int64_t edge_count = edges.GetLength();
	int64_t layer_count = layer_decimation_radii.GetLength();

	o3c::AssertTensorShape(node_positions, {node_count, 3});
	o3c::AssertTensorDtype(node_positions, o3c::Float32);
	o3c::AssertTensorDevice(node_positions, device);

	o3c::AssertTensorShape(node_rotations, {node_count, 3, 3});
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorDevice(node_rotations, device);

	o3c::AssertTensorShape(node_translations, {node_count, 3});
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorDevice(node_translations, device);

	o3c::AssertTensorShape(edges, {edge_count, 2});
	o3c::AssertTensorDtype(edges, o3c::Int32);
	o3c::AssertTensorDevice(edges, device);

	o3c::AssertTensorShape(edge_layer_indices, {edge_count});
	o3c::AssertTensorDtype(edge_layer_indices, o3c::Int32);
	o3c::AssertTensorDevice(edge_layer_indices, device);

	o3c::AssertTensorShape(layer_decimation_radii, {layer_count});
	o3c::AssertTensorDtype(layer_decimation_radii, o3c::Float32);
	o3c::AssertTensorDevice(layer_decimation_radii, device);

	// === prepare inputs
	auto node_position_data = node_positions.GetDataPtr<float>();
	auto node_rotation_data = node_rotations.GetDataPtr<float>();
	auto node_translation_data = node_translations.GetDataPtr<float>();
	auto edge_data = edges.GetDataPtr<int>();
	auto edge_layer_index_data = edge_layer_indices.GetDataPtr<int>();
	auto layer_decimation_radii_data = layer_decimation_radii.GetDataPtr<float>();

	// === prepare outputs
	edge_jacobians = o3c::Tensor({edge_count, 2, 3}, o3c::Float32, device);
	auto edge_jacobian_data = edge_jacobians.GetDataPtr<float>();
	node_edge_jacobian_indices_jagged = o3c::Tensor({node_count, MAX_EDGES_PER_NODE}, o3c::Int32, device);
	auto node_edge_jacobian_index_data = node_edge_jacobian_indices_jagged.GetDataPtr<int32_t>();

	core::AtomicCounterArray<TDeviceType> node_edge_jacobian_counters(node_count);

	// o3c::ParallelFor(
	// 		device,
	// 		);

}

} // namespace nnrt::alignment::functional::kernel