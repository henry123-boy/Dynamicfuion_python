//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
//  Copyright (c) 2022 Gregory Kramida
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
#include "rendering/functional/kernel/PartialDerivatives.h"
#include "core/PlatformIndependence.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::rendering::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType>
void WarpedVertexAndNormalJacobians(open3d::core::Tensor& vertex_jacobians, open3d::core::Tensor& normal_jacobians,
                                    const open3d::core::Tensor& vertex_positions, const open3d::core::Tensor& vertex_normals,
                                    const open3d::core::Tensor& node_positions, const open3d::core::Tensor& node_rotations,
                                    const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights) {
	auto device = vertex_positions.GetDevice();
	o3c::AssertTensorDevice(vertex_normals, device);
	o3c::AssertTensorDevice(node_positions, device);
	o3c::AssertTensorDevice(node_rotations, device);
	o3c::AssertTensorDevice(warp_anchors, device);
	o3c::AssertTensorDevice(warp_anchor_weights, device);
	o3c::AssertTensorShape(vertex_positions, { utility::nullopt, 3 });
	o3c::AssertTensorShape(vertex_normals, vertex_positions.GetShape());
	auto node_count = node_positions.GetLength();
	o3c::AssertTensorShape(node_rotations, {node_count, 3, 3});
	auto vertex_count = vertex_positions.GetLength();
	o3c::AssertTensorShape(warp_anchors, {vertex_count, utility::nullopt});
	o3c::AssertTensorShape(warp_anchor_weights, warp_anchors.GetShape());
	auto anchors_per_vertex = warp_anchors.GetShape(1);
	o3c::AssertTensorDtype(warp_anchors,)

	const auto*


	vertex_jacobians = o3c::Tensor({vertex_count, anchors_per_vertex, 3, 3}, vertex_positions.GetDtype(), device);
	normal_jacobians = o3c::Tensor({vertex_count, anchors_per_vertex, 3, 3}, vertex_positions.GetDtype(), device);

	o3c::ParallelFor(device, vertex_count * anchors_per_vertex,
	                 NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		const auto i_vertex = workload_idx / anchors_per_vertex;
		const auto i_anchor = workload_idx % anchors_per_vertex;
		const auto
	});

}

} // namespace nnrt::rendering::functional::kernel

