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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/utility/Optional.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <core/PlatformIndependentArray.h>

// local includes
namespace nnrt::rendering::kernel {
void ShadeEdgesFlat(
        open3d::core::Tensor &pixels,
        const open3d::core::Tensor &pixel_face_indices,
        const open3d::core::Tensor &pixel_depths,
        const open3d::core::Tensor &pixel_barycentric_coordinates,
        const open3d::core::Tensor &pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes,
        float ndc_line_width,
        const std::array<float, 3>& color
);

template<open3d::core::Device::DeviceType TDeviceType>
void ShadeEdgesFlat(
        open3d::core::Tensor &pixels,
        const open3d::core::Tensor &pixel_face_indices,
        const open3d::core::Tensor &pixel_depths,
        const open3d::core::Tensor &pixel_barycentric_coordinates,
        const open3d::core::Tensor &pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes,
        float ndc_line_width,
        const std::array<float, 3>& color
);

} // namespace nnrt::rendering::kernel