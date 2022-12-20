//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/12/22.
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

// 3rd party
#include <open3d/geometry/TriangleMesh.h>

// local
#include "geometry.h"
#include "geometry/functional/NormalsOperations.h"


namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

namespace test {

open3d::t::geometry::TriangleMesh
GenerateXyPlane(
        float plane_side_length, const std::tuple<float, float, float>& plane_center_position, int subdivision_count,
        const open3d::core::Device& device
) {
    auto mesh = open3d::t::geometry::TriangleMesh(device);

    float hsl = plane_side_length / 2.f;
    mesh.SetVertexPositions(
            o3c::Tensor(std::vector<float>{
                    -hsl, -hsl, 0.f,
                    -hsl, hsl, 0.f,
                    hsl, -hsl, 0.f,
                    hsl, hsl, 0.f
            }, {4, 3}, o3c::Float32, device)
            +
            o3c::Tensor(std::vector<float>{
                    std::get<0>(plane_center_position),
                    std::get<1>(plane_center_position),
                    std::get<2>(plane_center_position)
            }, {1, 3}, o3c::Float32, device)
    );
    mesh.SetTriangleIndices(o3c::Tensor(std::vector<int64_t>{
            0, 1, 2,
            2, 1, 3
    }, {2, 3}, o3c::Int64, device));
    mesh.SetTriangleNormals(o3c::Tensor(std::vector<float>{
            0, 0, -1,
            0, 0, -1
    }, {2, 3}, o3c::Float32, device));
    mesh.SetVertexNormals(o3c::Tensor(std::vector<float>{
            0, 0, -1,
            0, 0, -1,
            0, 0, -1,
            0, 0, -1
    }, {4, 3}, o3c::Float32, device));
    mesh.SetVertexColors(o3c::Tensor::Full({4, 3}, 0.7f, o3c::Float32, device));

    if (subdivision_count > 0) {
        auto mesh_legacy = mesh.ToLegacy().SubdivideMidpoint(subdivision_count);
        return o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);
    }
    return mesh;
}

open3d::t::geometry::TriangleMesh GenerateSphere(
        float radius,
        const std::tuple<float, float, float>& center,
        int resolution,
        const open3d::core::Device& device
) {
    auto mesh_legacy = open3d::geometry::TriangleMesh::CreateSphere(radius, resolution);
    auto mesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);
    mesh.SetVertexPositions(mesh.GetVertexPositions() + o3c::Tensor(std::vector<float>{
                    std::get<0>(center),
                    std::get<1>(center),
                    std::get<2>(center)
            }, {1, 3}, o3c::Float32, device));

    nnrt::geometry::ComputeVertexNormals(mesh);
    mesh.SetVertexColors(o3c::Tensor::Full({mesh.GetVertexPositions().GetLength(), 3}, 0.7f, o3c::Float32, device));

    return mesh;
}

void SetMeshColorGreyscaleValue(open3d::t::geometry::TriangleMesh& mesh, float greyscale_value){
    auto vertex_count = mesh.GetVertexPositions().GetLength();
	mesh.SetVertexColors(o3c::Tensor::Full({vertex_count, 3}, greyscale_value, o3c::Float32, mesh.GetDevice()));
}

void OffsetMesh(open3d::t::geometry::TriangleMesh& mesh, const std::tuple<float, float, float>& offset){
	mesh.SetVertexPositions(mesh.GetVertexPositions() +
	                        o3c::Tensor(std::vector<float>{
			                                    std::get<0>(offset),
			                                    std::get<1>(offset),
			                                    std::get<2>(offset),
	                                    },
	                                    {3}, o3c::Float32, mesh.GetDevice())
	);
}

} // namespace test