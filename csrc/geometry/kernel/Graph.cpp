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
#include "geometry/kernel/Graph.h"

using namespace open3d;

namespace nnrt::geometry::kernel::graph {
void ComputeAnchorsAndWeightsEuclidean(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                       const open3d::core::Tensor& nodes, const int anchor_count, const int minimum_valid_anchor_count,
                                       const float node_coverage) {
	core::Device device = points.GetDevice();
	core::Device::DeviceType device_type = device.GetType();

	switch (device_type) {
		case core::Device::DeviceType::CPU:
			if (minimum_valid_anchor_count > 0) {
				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CPU, true>(
						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
			} else {
				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CPU, false>(
						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
			}
			break;
		case core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			if (minimum_valid_anchor_count > 0) {
				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CUDA, true>(
						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
			} else {
				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CUDA, false>(
						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
			}
#else
			utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			utility::LogError("Unimplemented device");
			break;
	}
}

void ComputeAnchorsAndWeightsShortestPath(core::Tensor& anchors, core::Tensor& weights, const core::Tensor& points, const core::Tensor& nodes,
                                          const core::Tensor& edges, int anchor_count, float node_coverage) {
	core::Device device = points.GetDevice();
	core::Device::DeviceType device_type = device.GetType();

	switch (device_type) {
		case core::Device::DeviceType::CPU:
			ComputeAnchorsAndWeightsShortestPath<core::Device::DeviceType::CPU>(
					anchors, weights, points, nodes, edges, anchor_count, node_coverage);
			break;
		case core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			ComputeAnchorsAndWeightsShortestPath<core::Device::DeviceType::CUDA>(
					anchors, weights, points, nodes, edges, anchor_count, node_coverage);

#else
			utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			utility::LogError("Unimplemented device");
			break;
	}
}

} // namespace nnrt::geometry::kernel::graph