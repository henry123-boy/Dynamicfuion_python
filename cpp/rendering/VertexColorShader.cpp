//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/21/22.
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
#include "rendering/VertexColorShader.h"
#include "rendering/functional/InterpolateVertexAttributes.h"

namespace utility = open3d::utility;
namespace o3c = open3d::core;
namespace nnrt::rendering {
VertexColorShader::VertexColorShader() {}

open3d::t::geometry::Image VertexColorShader::ShadeMeshes(
        const open3d::core::Tensor& pixel_face_indices,
        const open3d::core::Tensor& pixel_depths,
        const open3d::core::Tensor& pixel_barycentric_coordinates,
        const open3d::core::Tensor& pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes
) const {

    if (!meshes.has_value() || meshes.value().get().empty()) {
        utility::LogError("VertexColorShader requires meshes argument to be filled with meshes that were used during "
                          "rasterization resulting in its other arguments, in order to retrieve and interpolate the "
                          "vertex colors.");
    }
    const std::vector<open3d::t::geometry::TriangleMesh>& source_meshes = meshes.value().get();
    o3c::Tensor vertex_colors = source_meshes[0].GetVertexColors();
    auto triangle_indices = source_meshes[0].GetTriangleIndices();
	auto face_vertex_colors = vertex_colors.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));
    o3c::Tensor cumulative_face_vertex_colors = face_vertex_colors;
    for (int i_mesh = 1; i_mesh < static_cast<int>(source_meshes.size()); i_mesh++) {
        vertex_colors = source_meshes[i_mesh].GetVertexColors();
        triangle_indices = source_meshes[i_mesh].GetTriangleIndices();
        face_vertex_colors = vertex_colors.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));
        cumulative_face_vertex_colors = cumulative_face_vertex_colors.Append(face_vertex_colors);
    }
    int64_t image_height = pixel_face_indices.GetShape(0);
    int64_t image_width = pixel_face_indices.GetShape(1);
    open3d::core::Tensor pixels =
            (functional::InterpolateVertexAttributes(pixel_face_indices, pixel_barycentric_coordinates,
                                                     cumulative_face_vertex_colors)
                     .Slice(2, 0, 1).Reshape({image_height, image_width, 3}) * 255.f).To(o3c::UInt8);

    return {pixels};
}
} // namespace nnrt::rendering