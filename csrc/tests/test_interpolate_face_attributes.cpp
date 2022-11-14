//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/19/22.
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
// test framework
#include "test_main.hpp"
#include "tests/test_utils/test_utils.hpp"

// 3rd parth
#include <open3d/t/io/TriangleMeshIO.h>

// code being tested
#include "geometry/NormalsOperations.h"
#include "rendering/functional/InterpolateFaceAttributes.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;

void TestInterpolateFaceAttributes(const o3c::Device& device, const std::string& mesh_name) {
	o3tg::TriangleMesh mesh;
	o3tio::ReadTriangleMesh(test::generated_mesh_test_data_directory.ToString() + "/" + mesh_name + ".ply", mesh);
	mesh = mesh.To(device);
	nnrt::geometry::ComputeVertexNormals(mesh);

	auto vertex_normals = mesh.GetVertexNormals();
	auto triangle_indices = mesh.GetTriangleIndices();
	auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));

	auto pixel_face_indices = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_face_indices.npy"
	).To(device);
	auto pixel_barycentric_coordinates = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_barycentric_coordinates.npy").To(device);

	auto interpolated_normals = nnrt::rendering::functional::InterpolateFaceAttributes(pixel_face_indices, pixel_barycentric_coordinates, face_vertex_normals);
	auto ground_truth_normals = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_rendered_normals.npy"
	).To(device);

	REQUIRE(interpolated_normals.AllClose(ground_truth_normals));

}

TEST_CASE("Test Interpolate Face Attributes - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInterpolateFaceAttributes(device, "mesh_bunny_res4");
}