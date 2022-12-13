//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/13/22.
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
// 3rd-party
#include <open3d/t/geometry/TriangleMesh.h>

// local
#include "alignment/FlatTriangleFitter.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

inline
o3tg::TriangleMesh MeshFrom2dTriangle(const Matrix3x2f& subject_triangle, const o3c::Device& device, float depth) {
    o3tg::TriangleMesh mesh(device);
    std::vector<float> triangle_vertices = {
            subject_triangle(0,0), subject_triangle(0,1), depth,
            subject_triangle(1,0), subject_triangle(1,1), depth,
            subject_triangle(2,0), subject_triangle(2,1), depth
    };

    mesh.SetVertexPositions(o3c::Tensor(triangle_vertices, {3,3}, o3c::Float32, device));
    mesh.SetTriangleIndices(o3c::Tensor(std::vector<int64_t>({0, 1, 2}), {1, 3}, o3c::Int64, device));
    return mesh;
}



std::vector<open3d::t::geometry::Image> FlatTriangleFitter::FitFlatTriangles(
        const Matrix3x2f& subject_triangle,
        const Matrix3x2f& reference_triangle,
        open3d::core::Device device
) {
    o3u::LogError("Not implemented.");
    std::vector<open3d::t::geometry::Image> iteration_shots;
    return iteration_shots;
}

} // namespace nnrt::alignment