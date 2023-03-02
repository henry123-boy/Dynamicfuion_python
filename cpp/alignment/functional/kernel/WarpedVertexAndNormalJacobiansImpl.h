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
#include <Eigen/Dense>


// local includes
#include "alignment/functional/kernel/WarpedVertexAndNormalJacobians.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::alignment::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType>
void WarpedVertexAndNormalJacobians(
        open3d::core::Tensor& vertex_position_jacobians, open3d::core::Tensor& vertex_normal_jacobians,
        const open3d::core::Tensor& vertex_positions, const open3d::core::Tensor& vertex_normals,
        const open3d::core::Tensor& node_positions, const open3d::core::Tensor& node_rotations,
        const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights
) {
    auto device = vertex_positions.GetDevice();
    o3c::AssertTensorDevice(vertex_normals, device);
    o3c::AssertTensorDevice(node_positions, device);
    o3c::AssertTensorDevice(node_rotations, device);
    o3c::AssertTensorDevice(warp_anchors, device);
    o3c::AssertTensorDevice(warp_anchor_weights, device);
    o3c::AssertTensorShape(vertex_positions, { utility::nullopt, 3 });
    o3c::AssertTensorShape(vertex_normals, vertex_positions.GetShape());
    auto node_count = node_positions.GetLength();
    o3c::AssertTensorShape(node_rotations, { node_count, 3, 3 });
    auto vertex_count = vertex_positions.GetLength();
    o3c::AssertTensorShape(warp_anchors, { vertex_count, utility::nullopt });
    o3c::AssertTensorShape(warp_anchor_weights, warp_anchors.GetShape());
    auto anchors_per_vertex = warp_anchors.GetShape(1);
    o3c::AssertTensorDtype(warp_anchors, o3c::Int32);
    o3c::AssertTensorDtype(warp_anchor_weights, o3c::Float32);
    o3c::AssertTensorDtype(vertex_positions, o3c::Float32);
    o3c::AssertTensorDtype(vertex_normals, o3c::Float32);
    o3c::AssertTensorDtype(node_positions, o3c::Float32);
    o3c::AssertTensorDtype(node_rotations, o3c::Float32);

    const auto* warp_anchor_data = warp_anchors.template GetDataPtr<int32_t>();
    const auto* warp_anchor_weight_data = warp_anchor_weights.template GetDataPtr<float>();

    const auto* vertex_position_data = vertex_positions.GetDataPtr<float>();
    const auto* vertex_normal_data = vertex_normals.GetDataPtr<float>();
    const auto* node_position_data = node_positions.GetDataPtr<float>();
    const auto* node_rotation_data = node_rotations.GetDataPtr<float>();

    // these will be used as skew-symmetric vectors later
	// TODO: potential optimization -- maybe can start with uninitialized-value tensors instead of Zeroes
    vertex_position_jacobians = o3c::Tensor::Zeros({vertex_count, anchors_per_vertex, 4}, vertex_positions.GetDtype(), device);
    vertex_normal_jacobians = o3c::Tensor::Zeros({vertex_count, anchors_per_vertex, 3}, vertex_positions.GetDtype(), device);

    auto* vertex_position_jacobian_data = vertex_position_jacobians.GetDataPtr<float>();
    auto* vertex_normal_jacobian_data = vertex_normal_jacobians.GetDataPtr<float>();

    o3c::ParallelFor(
            device, vertex_count * anchors_per_vertex,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
                const auto i_vertex = workload_idx / anchors_per_vertex;
                const auto i_anchor = workload_idx % anchors_per_vertex;
                const auto i_node = warp_anchor_data[i_vertex * anchors_per_vertex + i_anchor];
				if(i_node == -1){
					return; // sentinel value;
				}
                const auto node_weight = warp_anchor_weight_data[i_vertex * anchors_per_vertex + i_anchor];
                Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>
                        node_rotation(node_rotation_data + (i_node * 9));
                Eigen::Map<const Eigen::Vector3f> node_position(node_position_data + (i_node * 3));
                Eigen::Map<const Eigen::Vector3f> vertex_position(vertex_position_data + (i_vertex * 3));
                Eigen::Map<const Eigen::Vector3f> vertex_normal(vertex_normal_data + (i_vertex * 3));

                Eigen::Map<Eigen::Vector3f> vertex_rotation_jacobian
                        (vertex_position_jacobian_data + (i_vertex * anchors_per_vertex * 4) + (i_anchor * 4));
                // store node weight to increase speed of retrieval for the vertex translation jacobian
                float* stored_node_weight =
                        vertex_position_jacobian_data + (i_vertex * anchors_per_vertex * 4) + (i_anchor * 4) + 3;
                *stored_node_weight = -node_weight;
                Eigen::Map<Eigen::Vector3f> normal_rotation_jacobian
                        (vertex_normal_jacobian_data + (i_vertex * anchors_per_vertex * 3) + (i_anchor * 3));

                vertex_rotation_jacobian = -node_weight * (node_rotation * (vertex_position - node_position));
                normal_rotation_jacobian = -node_weight * (node_rotation * vertex_normal);
            }
    );

}


} // namespace nnrt::alignment::functional::kernel

