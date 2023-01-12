//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/5/23.
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
// stdlib includes

// third-party includes

// local includes
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/DeviceSelection.h"

namespace nnrt::alignment::kernel {

void ComputePixelVertexAnchorJacobiansAndNodeAssociations(
        open3d::core::Tensor& pixel_vertex_anchor_jacobians,
        open3d::core::Tensor& node_pixel_vertex_jacobians,
        open3d::core::Tensor& node_pixel_vertex_jacobian_counts,
        const open3d::core::Tensor& rasterized_vertex_position_jacobians,
        const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
        const open3d::core::Tensor& warped_vertex_position_jacobians,
        const open3d::core::Tensor& warped_vertex_normal_jacobians,
        const open3d::core::Tensor& point_map_vectors,
        const open3d::core::Tensor& rasterized_normals,
        const open3d::core::Tensor& residual_mask,
        const open3d::core::Tensor& pixel_faces,
        const open3d::core::Tensor& face_vertices,
        const open3d::core::Tensor& vertex_anchors,
        int64_t node_count
) {
    core::ExecuteOnDevice(
            residual_mask.GetDevice(),
            [&] {
                ComputePixelVertexAnchorJacobiansAndNodeAssociations<open3d::core::Device::DeviceType::CPU>(
                        pixel_vertex_anchor_jacobians, node_pixel_vertex_jacobians, node_pixel_vertex_jacobian_counts,
                        rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
                        warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
                        point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_vertices,
                        vertex_anchors,
                        node_count);
            },
            [&] {
                NNRT_IF_CUDA(
                        ComputePixelVertexAnchorJacobiansAndNodeAssociations<open3d::core::Device::DeviceType::CUDA>(
                                pixel_vertex_anchor_jacobians, node_pixel_vertex_jacobians,
                                node_pixel_vertex_jacobian_counts,
                                rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
                                warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
                                point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_vertices,
                                vertex_anchors,
                                node_count);
                );
            }
    );
}

void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
        open3d::core::Tensor& node_jacobians,
        open3d::core::Tensor& node_jacobian_ranges,
        open3d::core::Tensor& node_jacobian_pixels,
        open3d::core::Tensor& node_pixel_vertex_jacobians,
        const open3d::core::Tensor& node_pixel_vertex_jacobian_counts,
        const open3d::core::Tensor& pixel_vertex_anchor_jacobians
) {
    core::ExecuteOnDevice(
            node_pixel_vertex_jacobians.GetDevice(),
            [&] {
                ConvertPixelVertexAnchorJacobiansToNodeJacobians < open3d::core::Device::DeviceType::CPU > (
                        node_jacobians, node_jacobian_ranges, node_jacobian_pixels,
                                node_pixel_vertex_jacobians, node_pixel_vertex_jacobian_counts, pixel_vertex_anchor_jacobians
                );
            },
            [&] {
                NNRT_IF_CUDA(
                        ConvertPixelVertexAnchorJacobiansToNodeJacobians < open3d::core::Device::DeviceType::CUDA > (
                                node_jacobians, node_jacobian_ranges, node_jacobian_pixels,
                                        node_pixel_vertex_jacobians, node_pixel_vertex_jacobian_counts, pixel_vertex_anchor_jacobians
                        );
                );
            }
    );
}

} // namespace nnrt::alignment::kernel