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

void ComputeHessianApproximation_BlockDiagonal(
        open3d::core::Tensor& pixel_jacobians,
        open3d::core::Tensor& node_pixel_lists,
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
                ComputeHessianApproximation_BlockDiagonal<open3d::core::Device::DeviceType::CPU>(
                        pixel_jacobians, node_pixel_lists,
                        rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
                        warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
                        point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_vertices,
                        vertex_anchors,
                        node_count);
			},
			[&] {
				NNRT_IF_CUDA(
                        ComputeHessianApproximation_BlockDiagonal<open3d::core::Device::DeviceType::CUDA>(
                                pixel_jacobians, node_pixel_lists,
                                rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
                                warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
                                point_map_vectors, rasterized_normals, residual_mask, pixel_faces, face_vertices,
                                vertex_anchors,
                                node_count);
				);
			}
	);
}
} // namespace nnrt::alignment::kernel