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
#include "JoinTriangleMeshes.h"
#include "core/TensorManipulationRoutines.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace utility = open3d::utility;


namespace nnrt::geometry::functional {


open3d::t::geometry::TriangleMesh JoinTriangleMeshes(
        const std::vector<open3d::t::geometry::TriangleMesh>& meshes
) {
    if (meshes.empty()) {
        utility::LogError("Need at least one mesh in the vector argument.");
    }
    const auto& first_mesh = meshes[0];
    o3c::Device device = first_mesh.GetDevice();
    size_t triangle_attribute_count = first_mesh.GetTriangleAttr().size();
    size_t vertex_attribute_count = first_mesh.GetVertexAttr().size();

    int i_mesh = 0;
    int mesh_count = static_cast<int>(meshes.size());

    for (const auto& mesh: meshes) {
        if (mesh.GetDevice() != device) {
            utility::LogError("Mesh devices must match! Got {} for mesh at index 0 and {} for mesh at index {}.",
                              device.ToString(), mesh.GetDevice().ToString(), i_mesh);
        }
        if (mesh.GetTriangleAttr().size() != triangle_attribute_count) {
            utility::LogError("Mesh triangle attributes must match! "
                              "Got {} attributes for mesh at index 0 and {} for mesh at index {}.",
                              triangle_attribute_count, mesh.GetTriangleAttr().size(), i_mesh);
        }
        if (mesh.GetVertexAttr().size() != vertex_attribute_count) {
            utility::LogError("Mesh vertex attributes must match! "
                              "Got {} attributes for mesh at index 0 and {} for mesh at index {}.",
                              vertex_attribute_count, mesh.GetVertexAttr().size(), i_mesh);
        }
        i_mesh++;
    }
    o3tg::TriangleMesh combined_mesh(device);


    const o3tg::TensorMap& first_mesh_vertex_attributes = first_mesh.GetVertexAttr();
    std::vector<int64_t> vertex_counts;

    for (const auto& kv: first_mesh_vertex_attributes) {
        o3c::Tensor combined_attribute = kv.second.Clone();
        if (kv.first == "positions") {
            vertex_counts.push_back(kv.second.GetLength());
        }
        for (i_mesh = 1; i_mesh < mesh_count; i_mesh++) {
            const o3tg::TriangleMesh& mesh = meshes[i_mesh];
            if (!mesh.GetVertexAttr().Contains(kv.first)) {
                utility::LogError("Meshes need to have matching vertex attributes. Got mesh at index 0 with {}, "
                                  "mesh at index {} without.", kv.first, i_mesh);
            } else {
                const o3c::Tensor& attribute_values = mesh.GetVertexAttr(kv.first);
                if (kv.first == "positions") {
                    vertex_counts.push_back(vertex_counts[i_mesh - 1] + attribute_values.GetLength());
                }
                combined_attribute = combined_attribute.Append(attribute_values, 0);
            }
        }
        combined_mesh.SetVertexAttr(kv.first, combined_attribute);
    }


    const o3tg::TensorMap& first_mesh_triangle_attributes = first_mesh.GetTriangleAttr();

    for (const auto& kv: first_mesh_triangle_attributes) {
        o3c::Tensor combined_attribute = kv.second.Clone();

        for (i_mesh = 1; i_mesh < mesh_count; i_mesh++) {
            const o3tg::TriangleMesh& mesh = meshes[i_mesh];
            if (!mesh.GetTriangleAttr().Contains(kv.first)) {
                utility::LogError("Meshes need to have matching triangle attributes. Got mesh at index 0 with {}, "
                                  "mesh at index {} without.", kv.first, i_mesh);
            } else {
                const o3c::Tensor& attribute_values = mesh.GetTriangleAttr(kv.first);
                if (kv.first == "indices") {
                    combined_attribute =
                            combined_attribute.Append(attribute_values +
                                                      core::SingleValueTensor(vertex_counts[i_mesh - 1], device), 0);
                } else {
                    combined_attribute = combined_attribute.Append(attribute_values, 0);
                }

            }
        }
        combined_mesh.SetTriangleAttr(kv.first, combined_attribute);
    }


    return combined_mesh;
}

} // namespace nnrt::geometry::functional
