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

// third-party
#include <open3d/t/geometry/Image.h>
#include <open3d/t/geometry/TriangleMesh.h>

// local
#include "core/platform_independence/Array.h"

namespace nnrt::rendering {

class FlatEdgeShader {

public:
    FlatEdgeShader(
            float pixel_line_width_,
            const std::array<float, 3>& line_color_
    );

    open3d::t::geometry::Image ShadeMeshes(
            const open3d::core::Tensor &pixel_face_indices,
            const open3d::core::Tensor &pixel_depths,
            const open3d::core::Tensor &pixel_barycentric_coordinates,
            const open3d::core::Tensor &pixel_face_distances,
            open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes
    ) const;

    void SetPixelLineWidth(float width);

    float GetNdcLineWidth(int image_height, int image_width) const;

    void SetLineColor(const std::array<float, 3> &color);

private:
    float pixel_line_width;
    std::array<float, 3> line_color;
};

} // nnrt::rendering
