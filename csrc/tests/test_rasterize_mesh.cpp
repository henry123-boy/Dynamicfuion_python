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
#include <open3d/t/io/TriangleMeshIO.h>
#include <open3d/t/io/ImageIO.h>

// code being tested
#include "rendering/RasterizeMesh.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3io = open3d::io;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;

bool AllMismatchesCloseToSegment(const Eigen::Vector2f& segment_start, const Eigen::Vector2f& segment_end,
                                 const o3c::Tensor& mismatch_locations, float max_distance = 0.2) {
	o3c::Tensor bad_points = mismatch_locations.Slice(0, 0, 2).Contiguous().To(o3c::Float32);
	Eigen::Vector2f segment_direction = (segment_end - segment_start).normalized().eval();
	o3c::Tensor segment_direction_o3d(segment_direction.data(), {1, 2}, o3c::Float32, mismatch_locations.GetDevice());
	o3c::Tensor segment_start_o3d(segment_start.data(), {2, 1}, o3c::Float32, mismatch_locations.GetDevice());
	o3c::Tensor closest_diagonal_stops = segment_direction_o3d.Matmul(bad_points) - segment_direction_o3d.Matmul(segment_start_o3d);
	o3c::Tensor closest_diagonal_points = segment_start_o3d + closest_diagonal_stops.Append(closest_diagonal_stops, 0) * segment_direction_o3d.T();

	o3c::Tensor vectors = (bad_points - closest_diagonal_points);
	vectors *= vectors;
	o3c::Tensor distances = vectors.Sum({0});
	return distances.AllClose(o3c::Tensor::Zeros(distances.GetShape(), o3c::Float32, distances.GetDevice()), 0.f, max_distance);
}

void TestRasterizeMeshNaive_Plane_NoMaskExtraction(const o3c::Device& device) {
	auto plane = test::GenerateXyPlane(1.0, std::make_tuple(0.f, 0.f, 2.f), 0, device);
	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::SizeVector image_size{480, 640};

	auto extracted_face_vertices = nnrt::rendering::ExtractClippedFaceVerticesInNormalizedCameraSpace(plane, intrinsics, {480, 640}, 0.0, 2.0);

	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			nnrt::rendering::RasterizeMesh(extracted_face_vertices, o3u::nullopt, image_size, 0.f, 1, 0, 0, false, false, true);

	auto pixel_face_indices_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_face_indices.npy").To(device);

	auto pixel_depths_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_depths.npy").To(device);

	auto pixel_barycentric_coordinates_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_barycentric_coordinates.npy").To(device);

	auto pixel_face_distances_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_face_distances.npy").To(device);

	auto mismatches_face_indices = (pixel_face_indices.IsClose(pixel_face_indices_ground_truth)).LogicalNot();

	auto face_index_mismatch_locations = mismatches_face_indices.NonZero();

	// allow discrepancies only around diagonal
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), face_index_mismatch_locations));
	auto pixel_depth_mismatch_locations = pixel_depths.IsClose(pixel_depths_ground_truth).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), pixel_depth_mismatch_locations));
	auto pixel_barycentric_coordinate_mismatch_locations =
			pixel_barycentric_coordinates.IsClose(pixel_barycentric_coordinates_ground_truth, 1e-5, 1e-6).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f),
	                                    pixel_barycentric_coordinate_mismatch_locations));

	auto pixel_face_distance_mismatch_locations = pixel_face_distances.IsClose(pixel_face_distances_ground_truth).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), pixel_face_distance_mismatch_locations));
}

TEST_CASE("Test Rasterize Mesh Naive - Plane - No-Mask Extraction - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMeshNaive_Plane_NoMaskExtraction(device);
}

TEST_CASE("Test Rasterize Mesh Naive - Plane - No-Mask Extraction  CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive_Plane_NoMaskExtraction(device);
}

void TestRasterizeMeshNaive_Plane_MaskExtraction(const o3c::Device& device) {
	auto plane = test::GenerateXyPlane(1.0, std::make_tuple(0.f, 0.f, 2.f), 0, device);
	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::SizeVector image_size{480, 640};

	auto [extracted_face_vertices, clipped_face_mask] =
			nnrt::rendering::ExtracFaceVerticesAndClipMaskInNormalizedCameraSpace(plane, intrinsics, {480, 640}, 0.0, 2.0);

	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			nnrt::rendering::RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1, 0, 0, false, false, true);

	auto pixel_face_indices_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_face_indices.npy").To(device);

	auto pixel_depths_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_depths.npy").To(device);

	auto pixel_barycentric_coordinates_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_barycentric_coordinates.npy").To(device);

	auto pixel_face_distances_ground_truth = open3d::core::Tensor::Load(
			test::generated_array_test_data_directory.ToString() + "/plane_0_pixel_face_distances.npy").To(device);

	auto mismatches_face_indices = (pixel_face_indices.IsClose(pixel_face_indices_ground_truth)).LogicalNot();

	auto face_index_mismatch_locations = mismatches_face_indices.NonZero();

	// allow discrepancies only around diagonal
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), face_index_mismatch_locations));
	auto pixel_depth_mismatch_locations = pixel_depths.IsClose(pixel_depths_ground_truth).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), pixel_depth_mismatch_locations));
	auto pixel_barycentric_coordinate_mismatch_locations =
			pixel_barycentric_coordinates.IsClose(pixel_barycentric_coordinates_ground_truth, 1e-5, 1e-6).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f),
	                                    pixel_barycentric_coordinate_mismatch_locations));

	auto pixel_face_distance_mismatch_locations = pixel_face_distances.IsClose(pixel_face_distances_ground_truth).LogicalNot().NonZero();
	REQUIRE(AllMismatchesCloseToSegment(Eigen::Vector2f(95.f, 175.f), Eigen::Vector2f(340.f, 420.f), pixel_face_distance_mismatch_locations));
}

TEST_CASE("Test Rasterize Mesh Naive - Plane - Mask Extraction - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMeshNaive_Plane_MaskExtraction(device);
}

TEST_CASE("Test Rasterize Mesh Naive - Plane - Mask Extraction  CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive_Plane_MaskExtraction(device);
}

void TestRasterizeMeshNaive(
		const o3c::Device& device, const std::string& mesh_name,
		const std::tuple<float, float, float>& offset = std::make_tuple(0.f, 0.f, 1.f),
		float color_value = 1.0f, bool mesh_is_generated = true, bool save_output_to_disk = false
) {
	o3tg::TriangleMesh mesh;

	if (mesh_is_generated) {
		o3tio::ReadTriangleMesh(test::generated_mesh_test_data_directory.ToString() + "/" + mesh_name + ".ply", mesh);
		mesh = mesh.To(device);
	} else {
		o3tio::ReadTriangleMesh(test::mesh_test_data_directory.ToString() + "/" + mesh_name + ".ply", mesh);
		mesh = mesh.To(device);
	}

	auto vertex_count = mesh.GetVertexPositions().GetLength();
	mesh.SetVertexColors(o3c::Tensor::Full({vertex_count, 3}, color_value, o3c::Float32, device));
	mesh.SetVertexPositions(mesh.GetVertexPositions() +
	                        o3c::Tensor(std::vector<float>{
			                                    std::get<0>(offset),
			                                    std::get<1>(offset),
			                                    std::get<2>(offset),
	                                    },
	                                    {3}, o3c::Float32, device)
	);


	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::SizeVector image_size{480, 640};

	auto [extracted_face_vertices, clipped_face_mask]  =
			nnrt::rendering::ExtracFaceVerticesAndClipMaskInNormalizedCameraSpace(mesh, intrinsics, image_size, 0.0, 10.0);

	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			nnrt::rendering::RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1, 0, 0, false, false, true);

	if (save_output_to_disk) {
		pixel_face_indices.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_face_indices.npy");
		pixel_depths.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_depths.npy");
		pixel_barycentric_coordinates.Save(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_barycentric_coordinates.npy");
		pixel_face_distances.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_face_distances.npy");
	} else {
		auto pixel_face_indices_ground_truth = open3d::core::Tensor::Load(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_face_indices.npy"
		).To(device);

		auto pixel_depths_ground_truth = open3d::core::Tensor::Load(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_depths.npy").To(device);

		auto pixel_barycentric_coordinates_ground_truth = open3d::core::Tensor::Load(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_barycentric_coordinates.npy").To(device);

		auto pixel_face_distances_ground_truth = open3d::core::Tensor::Load(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_pixel_face_distances.npy").To(device);


		auto indices_mismatched = pixel_face_indices.IsClose(pixel_face_indices_ground_truth).LogicalNot();
		auto mismatched_face_indices_out = pixel_face_indices.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Flatten();
		auto mismatched_face_indices_gt = pixel_face_indices_ground_truth.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Flatten();
		auto mismatched_face_triangles_out = mesh.GetTriangleIndices().GetItem(o3c::TensorKey::IndexTensor(mismatched_face_indices_out.To(o3c::Int64)));
		auto mismatched_face_triangles_gt = mesh.GetTriangleIndices().GetItem(o3c::TensorKey::IndexTensor(mismatched_face_indices_gt.To(o3c::Int64)));
		// REQUIRE(mismatched_face_indices_out.Min({1}))

		std::cout << mismatched_face_triangles_out.Min({1}).Slice(0, 0, 10).ToString() << std::endl;
		std::cout << mismatched_face_triangles_gt.Min({1}).Slice(0, 0, 10).ToString() << std::endl;
		std::cout << mismatched_face_triangles_out.Max({1}).Slice(0, 0, 10).ToString() << std::endl;
		std::cout << mismatched_face_triangles_gt.Max({1}).Slice(0, 0, 10).ToString() << std::endl;


		// auto diff_image = o3tg::Image(diff.To(o3c::UInt8) * 255);
		// o3tio::WriteImage(test::generated_image_test_data_directory.ToString() + "/" + mesh_name + "_diff_mask.png", diff_image);


		// std::cout << indices_mismatched.NonZero().GetShape().ToString() << std::endl; // prints correct result for CPU and CUDA


		// int x_to_test = 175;
		// int y_to_test = 250;
		//
		// std::cout << pixel_face_indices[y_to_test][x_to_test].ToString() << std::endl;
		// std::cout << pixel_face_indices_ground_truth[y_to_test][x_to_test].ToString() << std::endl;


		// std::cout << pixel_face_indices.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Slice(0, 0, 10).ToString() << std::endl;
		// std::cout << pixel_face_indices_ground_truth.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Slice(0, 0, 10).ToString() << std::endl;

		// std::cout << pixel_face_distances.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Slice(0, 0, 10).ToString() << std::endl;
		// std::cout << pixel_face_distances_ground_truth.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Slice(0, 0, 10).ToString() << std::endl;

		// auto distance_mismatched = pixel_face_distances.IsClose(pixel_face_distances_ground_truth,1e-5,1e-9).LogicalNot();
		// std::cout << distance_mismatched.NonZero().GetShape().ToString() << std::endl;
		// std::cout << (pixel_face_distances_ground_truth - pixel_face_distances).GetItem(o3c::TensorKey::IndexTensor(indices_mismatched)).Abs().Max({0}).ToString() << std::endl;
		// std::cout << (pixel_face_distances_ground_truth - pixel_face_distances).GetItem(o3c::TensorKey::IndexTensor(distance_mismatched)).Abs().Max({0}).ToString() << std::endl;
		// std::cout << distance_mismatched.LogicalAnd(indices_mismatched).NonZero().GetShape().ToString() << std::endl;

		REQUIRE(pixel_face_distances.AllClose(pixel_face_distances_ground_truth));
		REQUIRE(pixel_depths_ground_truth.AllClose(pixel_depths_ground_truth));
		// REQUIRE(pixel_barycentric_coordinates.AllClose(pixel_barycentric_coordinates_ground_truth, 1e-5, 1e-6));
		// REQUIRE(pixel_face_indices.AllClose(pixel_face_indices_ground_truth));



	}

}

TEST_CASE("Test Rasterize Mesh Naive - Cube 0 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMeshNaive(device, "cube_0", std::make_tuple(0.f, 0.0f, 2.0f), 1.0, false, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMeshNaive(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 4 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 2 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive(device, "mesh_bunny_res2", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, false);
}

TEST_CASE("Test Rasterize Mesh Naive - 64 Bunnies - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMeshNaive(device, "mesh_64_bunny_array", std::make_tuple(0.f, 0.0f, 1.0f), 1.0, true, false);
}