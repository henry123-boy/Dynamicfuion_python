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
#include "WarpAnchorComputation.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::functional::kernel {


void ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		o3c::Tensor& anchors,
		o3c::Tensor& weights,
		const o3c::Tensor& points,
		const o3c::Tensor& nodes,
		const o3c::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
) {
	o3c::Device device = points.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				if (minimum_valid_anchor_count > 0) {
					ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight<o3c::Device::DeviceType::CPU, true>(
							anchors, weights, points, nodes, node_coverage_weights, anchor_count, minimum_valid_anchor_count);
				} else {
					ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight<o3c::Device::DeviceType::CPU, false>(
							anchors, weights, points, nodes, node_coverage_weights, anchor_count, minimum_valid_anchor_count);
				}
			},
			[&] {
				NNRT_IF_CUDA(
						if (minimum_valid_anchor_count > 0) {
							ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight<o3c::Device::DeviceType::CUDA, true>(
									anchors, weights, points, nodes, node_coverage_weights, anchor_count, minimum_valid_anchor_count);
						} else {
							ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight<o3c::Device::DeviceType::CUDA, false>(
									anchors, weights, points, nodes, node_coverage_weights, anchor_count, minimum_valid_anchor_count);
						}
				);
			}
	);
}

void ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
		float node_coverage
) {
	o3c::Device device = points.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				if (minimum_valid_anchor_count > 0) {
					ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight<o3c::Device::DeviceType::CPU, true>(
							anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
				} else {
					ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight<o3c::Device::DeviceType::CPU, false>(
							anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
				}
			},
			[&] {
				NNRT_IF_CUDA(
						if (minimum_valid_anchor_count > 0) {
							ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight<o3c::Device::DeviceType::CUDA, true>(
									anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
						} else {
							ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight<o3c::Device::DeviceType::CUDA, false>(
									anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
						}
				);
			}
	);
}

void ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
) {
	o3c::Device device = points.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight<o3c::Device::DeviceType::CPU>(
						anchors, weights, points, nodes, node_coverage_weights, edges, anchor_count);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight<o3c::Device::DeviceType::CUDA>(
								anchors, weights, points, nodes, node_coverage_weights, edges, anchor_count);
				);
			}
	);
}

void ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges, int anchor_count, float node_coverage
) {
	o3c::Device device = points.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight<o3c::Device::DeviceType::CPU>(
						anchors, weights, points, nodes, edges, anchor_count, node_coverage);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight<o3c::Device::DeviceType::CUDA>(
								anchors, weights, points, nodes, edges, anchor_count, node_coverage);
				);
			}
	);
}




} // namespace nnrt::geometry::functional::kernel