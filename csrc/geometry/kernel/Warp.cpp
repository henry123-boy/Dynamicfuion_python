//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/21.
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
#include "geometry/kernel/Warp.h"

using namespace open3d;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace warp {


void WarpPoints(open3d::core::Tensor& warped_vertices, const open3d::core::Tensor& anchors, const open3d::core::Tensor& weights,
                const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations,
                const open3d::core::Tensor& node_translations) {
	core::Device::DeviceType device_type = nodes.GetDevice().GetType();
	switch (device_type) {
		case core::Device::DeviceType::CPU:
			WarpPoints<core::Device::DeviceType::CPU>(
					warped_vertices, anchors, weights, points, nodes, node_rotations, node_translations);
			break;
		case core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			WarpPoints<core::Device::DeviceType::CUDA>(
					warped_vertices, anchors, weights, points, nodes, node_rotations, node_translations);
#else
			utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			utility::LogError("Unimplemented device");
			break;
	}
}


} // namespace warp
} // namespace kernel
} // namespace geometry
} // namespace nnrt