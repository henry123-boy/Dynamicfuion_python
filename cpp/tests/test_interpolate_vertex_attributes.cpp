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
#include "geometry/functional/NormalsOperations.h"
#include "rendering/functional/InterpolateVertexAttributes.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;

void TestInterpolateFaceAttributes(const o3c::Device& device, const std::string& mesh_name,
                                   const std::tuple<float, float, float>& offset = std::make_tuple(0.f, 0.f, 0.f),
								   int allowed_error_count = 0) {
	o3tg::TriangleMesh mesh;
	o3tio::ReadTriangleMesh(test::generated_mesh_test_data_directory.ToString() + "/" + mesh_name + ".ply", mesh);
	mesh = mesh.To(device);
	mesh.SetVertexPositions(mesh.GetVertexPositions() +
	                        o3c::Tensor(std::vector<float>{
			                                    std::get<0>(offset),
			                                    std::get<1>(offset),
			                                    std::get<2>(offset),
	                                    },
	                                    {3}, o3c::Float32, device)
	);
	nnrt::geometry::ComputeVertexNormals(mesh);

	auto vertex_normals = mesh.GetVertexNormals();
	auto triangle_indices = mesh.GetTriangleIndices();
	auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));

	auto pixel_face_indices = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_face_indices.npy"
	).To(device);
	auto pixel_barycentric_coordinates = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_barycentric_coordinates.npy").To(device);

	auto interpolated_normals =
            nnrt::rendering::functional::InterpolateVertexAttributes(pixel_face_indices, pixel_barycentric_coordinates,
                                                                     face_vertex_normals);
	auto ground_truth_normals = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_rendered_normals.npy"
	).To(device);

	int64_t image_height = interpolated_normals.GetShape(0);
	int64_t image_width = interpolated_normals.GetShape(1);
	int64_t faces_per_pixel = interpolated_normals.GetShape(2);
	int64_t pixel_attribute_count = interpolated_normals.GetShape(3);
	int64_t pixel_count = image_height * image_width;
	interpolated_normals = interpolated_normals.Reshape({pixel_count, faces_per_pixel, pixel_attribute_count});

	auto not_close = interpolated_normals.IsClose(ground_truth_normals, 1e-4,1e-5).LogicalNot();
	REQUIRE(not_close.NonZero().GetShape(1) <= allowed_error_count);

}

TEST_CASE("Test Interpolate Face Attributes - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInterpolateFaceAttributes(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 300);
}

TEST_CASE("Test Interpolate Face Attributes - Bunny Res 4 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInterpolateFaceAttributes(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 200);
}

TEST_CASE("Test Interpolate Face Attributes - Bunny Res 2 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInterpolateFaceAttributes(device, "mesh_bunny_res2", std::make_tuple(0.f, -0.1f, 0.3f), 200);
}

TEST_CASE("Test Interpolate Face Attributes - 64 Bunnies - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInterpolateFaceAttributes(device, "mesh_64_bunny_array", std::make_tuple(0.f, -0.1f, 0.3f), 3000);
}