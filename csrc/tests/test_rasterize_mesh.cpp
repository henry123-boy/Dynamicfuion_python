//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/19/22.
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
#include "tests/test_utils/geometry.h"
#include "tests/test_utils/test_utils.hpp"

// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/geometry/TriangleMesh.h>

// code being tested
#include "rendering/RasterizeMesh.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

void TestRasterizeMeshNaive(const o3c::Device& device){
	auto plane = test::GenerateXyPlane(1.0, std::make_tuple(0.f, 0.f, 2.f), 0, device);
	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::SizeVector image_size{480, 640};

	auto extracted_face_vertices = nnrt::rendering::ExtractClippedFaceVerticesInNormalizedCameraSpace(plane, intrinsics, {480, 640}, 0.0, 2.0);

	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			nnrt::rendering::RasterizeMesh(extracted_face_vertices, image_size, 0.f, 1, 0, 0, false, false, true);

	auto pixel_face_indices_ground_truth = open3d::core::Tensor::Load(
			test::array_test_data_directory.ToString() + "/plane_0_pixel_face_indices.npy").To(device);

	auto pixel_depths_ground_truth = open3d::core::Tensor::Load(
			test::array_test_data_directory.ToString() + "/plane_0_pixel_depths.npy").To(device);

	auto pixel_barycentric_coordinates_ground_truth = open3d::core::Tensor::Load(
			test::array_test_data_directory.ToString() + "/plane_0_pixel_barycentric_coordinates.npy").To(device);

	auto pixel_face_distances_ground_truth = open3d::core::Tensor::Load(
			test::array_test_data_directory.ToString() + "/plane_0_pixel_face_distances.npy").To(device);

	auto mismatches = (pixel_face_indices.IsClose(pixel_face_indices_ground_truth)).LogicalNot();
	// const int inspect_y = 200;
	// const int inspect_x = 350;
	// const int inspect_y = 300;
	// const int inspect_x = 250;
	const int inspect_y = 95;
	const int inspect_x = 175;
	std::cout << mismatches.To(o3c::Int32).Sum({0,1}).ToString() << std::endl;
	std::cout << mismatches.NonZero().Slice(1,0,30).ToString() << std::endl;

	std::cout << pixel_face_indices[inspect_y][inspect_x].ToString() << std::endl;
	std::cout << pixel_face_indices_ground_truth[inspect_y][inspect_x].ToString() << std::endl << std::endl;

	std::cout << pixel_depths[inspect_y][inspect_x].ToString() << std::endl;
	std::cout << pixel_depths_ground_truth[inspect_y][inspect_x].ToString() << std::endl << std::endl;

	std::cout << pixel_barycentric_coordinates[inspect_y][inspect_x].ToString() << std::endl;
	std::cout << pixel_barycentric_coordinates_ground_truth[inspect_y][inspect_x].ToString() << std::endl << std::endl;

	std::cout << pixel_face_distances[inspect_y][inspect_x].ToString() << std::endl;
	std::cout << pixel_face_distances_ground_truth[inspect_y][inspect_x].ToString() << std::endl << std::endl;

	// std::cout << pixel_face_indices.GetItem(o3c::TensorKey::IndexTensor(mismatches)).Slice(0,0,10).ToString() << std::endl << std::endl;
	// std::cout << pixel_face_indices_ground_truth.GetItem(o3c::TensorKey::IndexTensor(mismatches)).Slice(0,0,10).ToString() << std::endl;

	REQUIRE(pixel_face_indices.AllEqual(pixel_face_indices_ground_truth));
	REQUIRE(pixel_depths.AllEqual(pixel_depths_ground_truth));
	REQUIRE(pixel_barycentric_coordinates.AllEqual(pixel_barycentric_coordinates_ground_truth));
	REQUIRE(pixel_face_distances.AllEqual(pixel_face_distances_ground_truth));

}

TEST_CASE("Test Rasterize Mesh Naive - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMeshNaive(device);
}

TEST_CASE("Test Rasterize Mesh Naive - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive(device);
}