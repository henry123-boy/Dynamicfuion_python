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

namespace nnrt {
namespace geometry {
namespace kernel {
namespace graph {
void ComputeAnchorsAndWeightsEuclidean(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& vertices,
                                       const open3d::core::Tensor& nodes, const int anchor_count, const float node_coverage) {
	core::Device device = vertices.GetDevice();
	core::Device::DeviceType device_type = device.GetType();
	switch (device_type) {
		case core::Device::DeviceType::CPU:
			ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CPU>(
					anchors, weights, vertices, nodes, anchor_count, node_coverage);
			break;
		case core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CUDA>(
					anchors, weights, vertices, nodes, anchor_count,  node_coverage);
#else
			utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			utility::LogError("Unimplemented device");
			break;
	}
}
} // namespace graph
} // namespace kernel
} // namespace geometry
} // namespace nnrt