//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/12/22.
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
// stdlib includes

// third-party includes

// local includes
#include "rendering/kernel/EdgeShader.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::rendering::kernel {

void ShadeEdges(
        open3d::core::Tensor &pixels,
        const open3d::core::Tensor &pixel_face_indices,
        const open3d::core::Tensor &pixel_depths,
        const open3d::core::Tensor &pixel_barycentric_coordinates,
        const open3d::core::Tensor &pixel_face_distances,
        const open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes
) {
    o3c::Device device = pixel_face_indices.GetDevice();
    core::ExecuteOnDevice(
            device,
            [&] {
                ShadeEdges<o3c::Device::DeviceType::CPU>(pixels, pixel_face_indices, pixel_depths,
                                                         pixel_barycentric_coordinates,
                                                         pixel_face_distances, meshes);
            },
            [&] {
                NNRT_IF_CUDA(
                        ShadeEdges<o3c::Device::DeviceType::CUDA>(pixels, pixel_face_indices, pixel_depths,
                                                                  pixel_barycentric_coordinates,
                                                                  pixel_face_distances, meshes);
                );
            }
    );
}

} // namespace nnrt::rendering::kernel