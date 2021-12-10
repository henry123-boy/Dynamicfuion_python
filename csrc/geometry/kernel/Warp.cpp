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
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::warp {


void WarpPoints(open3d::core::Tensor& warped_vertices,
                const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations,
                const open3d::core::Tensor& node_translations, const int anchor_count, const float node_coverage) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				WarpPoints<o3c::Device::DeviceType::CPU>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage);
			},
			[&] {
				NNRT_IF_CUDA(WarpPoints<o3c::Device::DeviceType::CUDA>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage););
			}
	);
}

void WarpPoints(open3d::core::Tensor& warped_vertices,
                const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations,
                const open3d::core::Tensor& node_translations, const int anchor_count, const float node_coverage,
				const int minimum_valid_anchor_count) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				WarpPoints<o3c::Device::DeviceType::CPU>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage, minimum_valid_anchor_count);
			},
			[&] {
				NNRT_IF_CUDA(WarpPoints<o3c::Device::DeviceType::CUDA>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage, minimum_valid_anchor_count););
			}
	);
}

void WarpPoints(open3d::core::Tensor& warped_vertices,
                const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
                const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
				const int minimum_valid_anchor_count) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				WarpPoints<o3c::Device::DeviceType::CPU>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchors, anchor_weights, minimum_valid_anchor_count);
			},
			[&] {
				NNRT_IF_CUDA(WarpPoints<o3c::Device::DeviceType::CUDA>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchors, anchor_weights, minimum_valid_anchor_count););
			}
	);
}


} // namespace nnrt::geometry::kernel::warp