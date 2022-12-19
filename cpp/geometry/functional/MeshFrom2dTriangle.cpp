//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/19/22.
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
#include <open3d/core/Tensor.h>

#include <open3d/t/geometry/Utility.h>

// local includes
#include "MeshFrom2dTriangle.h"
#include "core/kernel/MathTypedefs.h"
#include "PerspectiveProjection.h"


namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry::functional {

o3tg::TriangleMesh MeshFrom2dTriangle(
        const open3d::core::Tensor& triangle_ndc,
        const o3c::Device& device,
        float depth,
        const o3c::Tensor& ndc_intrinsics
) {
    o3tg::CheckIntrinsicTensor(ndc_intrinsics);
    o3tg::TriangleMesh mesh(device);
    auto triangle_ndc_eigen = core::kernel::TensorToEigenMatrix<core::kernel::Matrix3x2f>(triangle_ndc);


    std::vector<float> projected_vertex_data = {
            triangle_ndc_eigen(0, 0), triangle_ndc_eigen(0, 1),
            triangle_ndc_eigen(1, 0), triangle_ndc_eigen(1, 1),
            triangle_ndc_eigen(2, 0), triangle_ndc_eigen(2, 1),
    };
    o3c::Tensor projected_vertices(projected_vertex_data, {3, 2}, o3c::Float32, device);
    o3c::Tensor projected_vertex_depth(std::vector<float>({depth}), {1}, o3c::Float32, device);

    o3c::Tensor vertex_positions =
            geometry::functional::UnprojectProjectedPoints(projected_vertices, projected_vertex_depth, ndc_intrinsics);

    core::kernel::Matrix3x2f triangle_vertex_normals;

    for(int i_previous_vertex = 0; i_previous_vertex < 3; i_previous_vertex++){
        int i_vertex = (i_previous_vertex + 1) % 3;
        int i_next_vertex = (i_vertex + 1) % 3;
        auto vertex = triangle_ndc_eigen.row(i_vertex);
        auto previous_vertex = triangle_ndc_eigen.row(i_previous_vertex);
        auto next_vertex = triangle_ndc_eigen.row(i_next_vertex);
        auto normal = ((vertex - previous_vertex).normalized() + (vertex - next_vertex).normalized()).normalized();
        triangle_vertex_normals.row(i_vertex) = normal;
    }

    std::vector<float> vertex_normals = {
            triangle_vertex_normals(0, 0), triangle_vertex_normals(0, 1), 0.f,
            triangle_vertex_normals(1, 0), triangle_vertex_normals(1, 1), 0.f,
            triangle_vertex_normals(2, 0), triangle_vertex_normals(2, 1), 0.f,
    };

    mesh.SetVertexPositions(vertex_positions);
    mesh.SetTriangleIndices(o3c::Tensor(std::vector<int64_t>({0, 1, 2}), {1, 3}, o3c::Int64, device));
    mesh.SetVertexNormals(o3c::Tensor(vertex_normals, {3, 3}, o3c::Float32, device));
    return mesh;
}

} // namespace nnrt::geometry::functional