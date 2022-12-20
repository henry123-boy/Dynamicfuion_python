//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/18/22.
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
#include "ExtractClippedFaceVertices.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::rendering::functional::kernel {


void MeshDataAndClippingMaskToNdc(
        open3d::core::Tensor& vertex_positions_normalized_camera,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_vertex_normals_camera,
        open3d::core::Tensor& clipped_face_mask,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_counts,
        const std::vector<open3d::core::Tensor>& vertex_positions_camera,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::core::Tensor>>> vertex_normals_camera,
        const std::vector<open3d::core::Tensor>& triangle_vertex_indices,
        const open3d::core::Tensor& normalized_camera_space_matrix,
        geometry::kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
        float near_clipping_distance,
        float far_clipping_distance
) {
    core::ExecuteOnDevice(
            vertex_positions_camera[0].GetDevice(),
            [&] {
                MeshDataAndClippingMaskToNdc<o3c::Device::DeviceType::CPU>(
                        vertex_positions_normalized_camera,
                        face_vertex_normals_camera,
                        clipped_face_mask, face_counts,
                        vertex_positions_camera,
                        vertex_normals_camera,
                        triangle_vertex_indices,
                        normalized_camera_space_matrix,
                        normalized_camera_space_xy_range,
                        near_clipping_distance,
                        far_clipping_distance
                );
            },
            [&] {
                NNRT_IF_CUDA(
                        MeshDataAndClippingMaskToNdc<o3c::Device::DeviceType::CUDA>(
                                vertex_positions_normalized_camera,
                                face_vertex_normals_camera,
                                clipped_face_mask,
                                face_counts,
                                vertex_positions_camera,
                                vertex_normals_camera,
                                triangle_vertex_indices,
                                normalized_camera_space_matrix,
                                normalized_camera_space_xy_range,
                                near_clipping_distance,
                                far_clipping_distance
                        );
                );
            }
    );
}
} // namespace nnrt::rendering::functional::kernel