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

// local
#include "FlatEdgeShader.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/kernel/FlatEdgeShader.h"

namespace nnrt::rendering {

FlatEdgeShader::FlatEdgeShader(
        float pixel_line_width_,
        const std::array<float, 3>& line_color_
) {
    SetPixelLineWidth(pixel_line_width_);
    this->line_color = line_color_;
}

open3d::t::geometry::Image FlatEdgeShader::ShadeMeshes(
        const open3d::core::Tensor &pixel_face_indices,
        const open3d::core::Tensor &pixel_depths,
        const open3d::core::Tensor &pixel_barycentric_coordinates,
        const open3d::core::Tensor &pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes
)
const {
    open3d::core::Tensor pixels;
    int64_t image_height = pixel_face_indices.GetShape(0);
    int64_t image_width = pixel_face_indices.GetShape(1);

    kernel::ShadeEdgesFlat(pixels, pixel_face_indices, pixel_depths, pixel_barycentric_coordinates,
                           pixel_face_distances,
                           meshes, this->GetNdcLineWidth(static_cast<int>(image_height), static_cast<int>(image_width)),
                           this->line_color);

    return {pixels};
}

void FlatEdgeShader::SetPixelLineWidth(float width) {
    this->pixel_line_width = width;
}

float FlatEdgeShader::GetNdcLineWidth(int image_height, int image_width) const {
    return kernel::ImageSpaceDistanceToNdc(this->pixel_line_width, image_height, image_width);
}

void FlatEdgeShader::SetLineColor(const std::array<float, 3> &color) {
    this->line_color = color;
}


} // nnrt::rendering