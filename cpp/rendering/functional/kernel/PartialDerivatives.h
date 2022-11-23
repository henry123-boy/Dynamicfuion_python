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
#include <open3d/core/Tensor.h>

// local includes

namespace nnrt::rendering::functional::kernel {
void WarpedVertexAndNormalJacobians(open3d::core::Tensor& vertex_rotation_jacobians, open3d::core::Tensor& normal_rotation_jacobians,
                                    const open3d::core::Tensor& vertex_positions, const open3d::core::Tensor& vertex_normals,
                                    const open3d::core::Tensor& node_positions, const open3d::core::Tensor& node_rotations,
                                    const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights);

template<open3d::core::Device::DeviceType TDeviceType>
void WarpedVertexAndNormalJacobians(open3d::core::Tensor& vertex_rotation_jacobians, open3d::core::Tensor& normal_rotation_jacobians,
                                    const open3d::core::Tensor& vertex_positions, const open3d::core::Tensor& vertex_normals,
                                    const open3d::core::Tensor& node_positions, const open3d::core::Tensor& node_rotations,
                                    const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights);


void RenderedVertexAndNormalJacobians(open3d::core::Tensor& rendered_vertex_jacobians, open3d::core::Tensor& rendered_normal_jacobians,
                                      const open3d::core::Tensor& warped_vertex_positions, const open3d::core::Tensor& warped_triangle_indices,
                                      const open3d::core::Tensor& warped_vertex_normals, const open3d::core::Tensor& pixel_faces,
                                      const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& ray_space_intrinsics);

template<open3d::core::Device::DeviceType TDeviceType>
void RenderedVertexAndNormalJacobians(open3d::core::Tensor& rendered_vertex_jacobians, open3d::core::Tensor& rendered_normal_jacobians,
                                      const open3d::core::Tensor& warped_vertex_positions, const open3d::core::Tensor& warped_triangle_indices,
                                      const open3d::core::Tensor& warped_vertex_normals, const open3d::core::Tensor& pixel_faces,
                                      const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& ray_space_intrinsics);

} // namespace nnrt::rendering::functional::kernel