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
#include "EdgeShader.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/kernel/EdgeShader.h"

namespace nnrt::rendering {

EdgeShader::EdgeShader(
        float pixel_line_width_,
        const open3d::core::SizeVector &rendered_image_size_,
        const nnrt::array<float, 3> line_color_
) {
    this->rendered_image_size = rendered_image_size_;
    SetPixelLineWidth(pixel_line_width_);
    this->line_color = line_color_;
}

open3d::t::geometry::Image EdgeShader::ShadeMeshes(
        const open3d::core::Tensor &pixel_face_indices,
        const open3d::core::Tensor &pixel_depths,
        const open3d::core::Tensor &pixel_barycentric_coordinates,
        const open3d::core::Tensor &pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes
)
const {
    open3d::core::Tensor pixels;
    kernel::ShadeEdges(pixels, pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances,
                       meshes);

    return open3d::t::geometry::Image(pixels);
}

void EdgeShader::SetPixelLineWidth(float width) {
    this->pixel_line_width = width;
    this->ndc_width = kernel::ImageSpaceDistanceToNdc(width, this->rendered_image_size[0],
                                                      this->rendered_image_size[1]);
}

void EdgeShader::SetRenderedImageSize(const open3d::core::SizeVector &size) {
    this->rendered_image_size = size;
    this->ndc_width = kernel::ImageSpaceDistanceToNdc(this->pixel_line_width, this->rendered_image_size[0],
                                                      this->rendered_image_size[1]);
}

const float &EdgeShader::GetNdcWidth() const {
    return this->ndc_width;
}

void EdgeShader::SetLineColor(const nnrt::array<float, 3> &color) {
    this->line_color = color;
}


} // nnrt::rendering