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
// stdlib
#include <algorithm>
#include <random>

// test framework
#include "test_main.hpp"
#include "tests/test_utils/geometry.h"
#include "tests/test_utils/test_utils.hpp"

// code being tested
#include "rendering/RasterizeMesh.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> VertexMat;

void TestExtractFaceVertices(const o3c::Device& device) {
	auto plane = test::GenerateXyPlane(1.2615, std::make_tuple(0.f, 0.f, 1.f), 4, device);
	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));

	auto extracted_face_vertices = nnrt::rendering::MeshFaceVerticesToRaySpace(plane, intrinsics, {480, 640}, 0.0, 2.0);

	REQUIRE(extracted_face_vertices.GetLength() == 334);

	auto extracted_face_vertices_ground_truth = open3d::core::Tensor::Load(
			test::array_test_data_directory.ToString() + "/extracted_face_vertices.npy").To(device);

	const int64_t count_faces_to_check = 10;
	std::random_device random_device{};
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<int64_t> random_distribution(0, extracted_face_vertices.GetLength()-1);
	for(int64_t i_test = 0; i_test < count_faces_to_check; i_test++){
		int64_t i_extracted_face = random_distribution(generator);
		bool found = false;
		auto face_vertices_normalized_camera = extracted_face_vertices[i_extracted_face];
		for(int64_t i_ground_truth_face = 0; i_ground_truth_face < extracted_face_vertices_ground_truth.GetLength() && !found; i_ground_truth_face++){
			auto face_vertices_ground_truth = extracted_face_vertices_ground_truth[i_ground_truth_face];
			found = face_vertices_normalized_camera.AllClose(face_vertices_ground_truth, 1e-5, 1e-7);
		}
		REQUIRE(found);
	}
}

TEST_CASE("Test Extract Face Vertices - CPU") {
	auto device = o3c::Device("CPU:0");
	TestExtractFaceVertices(device);
}

TEST_CASE("Test Extract Face Vertices - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestExtractFaceVertices(device);
}