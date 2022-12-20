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
#include <chrono>
#include <iomanip>

// code being tested
#include "rendering/RasterizeMesh.h"
#include "core/functional/Sorting.h"
#include "core/functional/Comparisons.h"
#include "rendering/functional/ExtractFaceVertices.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3io = open3d::io;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;

typedef open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> OptionalTensorWrapper;

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

void TestRasterizePlane_MaskExtraction(const o3c::Device& device, bool naive = true, bool save_output_to_disk = false) {
	auto plane = test::GenerateXyPlane(1.0, std::make_tuple(0.f, 0.f, 2.f), 0, device);
	o3c::Tensor intrinsics(std::vector<double>{
			580., 0., 320.,
			0., 580., 240.,
			0., 0., 1.,
	}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::SizeVector image_size{480, 640};

	auto [extracted_face_vertices, clipped_face_mask] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(plane, intrinsics, {480, 640}, 0.0, 2.0);

	int bin_size = -1;
	int max_faces_per_bin = 4;
	if (naive) {
		bin_size = max_faces_per_bin = 0;
	}
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			nnrt::rendering::RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1,
			                               bin_size, max_faces_per_bin, false, false, true);

	std::string mesh_name = "plane_0";
	if (save_output_to_disk) {
		pixel_face_indices.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_face_indices.npy");
		pixel_depths.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_depths.npy");
		pixel_barycentric_coordinates.Save(
				test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_barycentric_coordinates.npy");
		pixel_face_distances.Save(test::generated_array_test_data_directory.ToString() + "/" + mesh_name + "_out_pixel_face_distances.npy");
	} else {
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
}

TEST_CASE("Test Rasterize Plane Naive - Mask Extraction - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizePlane_MaskExtraction(device, true);
}

TEST_CASE("Test Rasterize Plane Naive - Mask Extraction  CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizePlane_MaskExtraction(device, true);
}

TEST_CASE("Test Rasterize Plane Coarse-to-Fine - Mask Extraction - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizePlane_MaskExtraction(device, false, false);
}

TEST_CASE("Test Rasterize Plane Coarse-to-Fine - Mask Extraction  CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizePlane_MaskExtraction(device, false, false);
}

void TestRasterizeMesh(
		const o3c::Device& device, const std::string& mesh_name,
		const std::tuple<float, float, float>& offset = std::make_tuple(0.f, 0.f, 1.f),
		float color_value = 1.0f, bool mesh_is_generated = true, int maximum_close_face_mismatches = 20,
		int maximum_close_face_mismatches_not_sharing_2_vertices = 0, bool naive = true,
		bool save_output_to_disk = false, bool print_benchmark = true
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

	int run_count = 1;
	if (print_benchmark) {
		run_count = 10;
	}

	auto start = std::chrono::high_resolution_clock::now();
	o3c::Tensor extracted_face_vertices, clipped_face_mask;
	for (int i_run = 0; i_run < run_count; i_run++){
		auto [extracted_face_vertices_local, clipped_face_mask_local] =
                nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(mesh, intrinsics, image_size, 0.0, 10.0);
		extracted_face_vertices = extracted_face_vertices_local;
		clipped_face_mask = clipped_face_mask_local;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	if (print_benchmark) {
		std::cout << "Average runtime of face vertex extraction & clip mask generation: " << std::setprecision(4) << (elapsed.count() * 1e-9)/run_count
		          << " seconds" << std::endl;
	}


	int bin_size = -1;
	int max_faces_per_bin = -1;
	if (naive) {
		bin_size = max_faces_per_bin = 0;
	}
	start = std::chrono::high_resolution_clock::now();
	o3c::Tensor pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances;
	if (print_benchmark) {
		auto [pixel_face_indices_local, pixel_depths_local, pixel_barycentric_coordinates_local, pixel_face_distances_local] =
				nnrt::rendering::RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1,
				                               bin_size, max_faces_per_bin, false, false, true);
		pixel_face_indices = pixel_face_indices_local;
		pixel_depths = pixel_depths_local;
		pixel_barycentric_coordinates = pixel_barycentric_coordinates_local;
		pixel_face_distances = pixel_face_distances_local;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	if (print_benchmark) {
		std::cout << "Average runtime of mesh rasterization: " << std::setprecision(4) << (elapsed.count() * 1e-9)/run_count << " seconds" << std::endl;
	}

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


		auto indices_mismatched_with_face_duplicates = pixel_face_indices.IsClose(pixel_face_indices_ground_truth).LogicalNot();
		// filter out duplicate faces, i.e. faces with exactly the mesh vertices (but, possibly, reordered) repeated in mesh
		auto mismatched_face_indices_with_face_duplicates_out =
				pixel_face_indices.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched_with_face_duplicates)).Flatten();
		auto mismatched_face_indices_with_face_duplicates_gt =
				pixel_face_indices_ground_truth.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched_with_face_duplicates)).Flatten();
		auto mismatched_face_triangles_with_face_duplicates_out =
				nnrt::core::functional::SortTensorAlongLastDimension(
						mesh.GetTriangleIndices().GetItem(
								o3c::TensorKey::IndexTensor(mismatched_face_indices_with_face_duplicates_out.To(o3c::Int64)))
				);
		auto mismatched_face_triangles_with_face_duplicates_gt =
				nnrt::core::functional::SortTensorAlongLastDimension(
						mesh.GetTriangleIndices().GetItem(o3c::TensorKey::IndexTensor(mismatched_face_indices_with_face_duplicates_gt.To(o3c::Int64)))
				);
		auto indices_mismatched_without_face_duplicates =
				mismatched_face_triangles_with_face_duplicates_out.IsClose(
						mismatched_face_triangles_with_face_duplicates_gt).LogicalNot().NonZero()[0];

		// check that number of mismatches is within reason and that they only differ by at most one vertex
		REQUIRE(indices_mismatched_without_face_duplicates.GetShape(0) < maximum_close_face_mismatches);
		auto mismatched_face_indices_without_face_duplicates_out =
				mismatched_face_indices_with_face_duplicates_out.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched_without_face_duplicates));
		auto mismatched_face_indices_without_face_duplicates_gt =
				mismatched_face_indices_with_face_duplicates_gt.GetItem(o3c::TensorKey::IndexTensor(indices_mismatched_without_face_duplicates));
		auto mismatched_face_triangles_without_face_duplicates_out =
				mesh.GetTriangleIndices().GetItem(o3c::TensorKey::IndexTensor(mismatched_face_indices_without_face_duplicates_out.To(o3c::Int64)));
		auto mismatched_face_triangles_without_face_duplicates_gt =
				mesh.GetTriangleIndices().GetItem(o3c::TensorKey::IndexTensor(mismatched_face_indices_without_face_duplicates_gt.To(o3c::Int64)));

		auto sharing_two_vertices = nnrt::core::functional::LastDimensionSeriesMatchUpToNElements(
				mismatched_face_triangles_without_face_duplicates_out, mismatched_face_triangles_without_face_duplicates_gt, 1
		);

		REQUIRE(sharing_two_vertices.NonZero().GetShape(1) <= maximum_close_face_mismatches_not_sharing_2_vertices);

		int64_t expected_mismatch_count = indices_mismatched_without_face_duplicates.GetShape(0);

		REQUIRE(pixel_face_distances.IsClose(pixel_face_distances_ground_truth).LogicalNot().NonZero().GetShape(1) <= expected_mismatch_count);
		REQUIRE(pixel_depths_ground_truth.AllClose(pixel_depths_ground_truth));
	}

}

TEST_CASE("Test Rasterize Mesh Naive - Cube 0 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMesh(device, "cube_0", std::make_tuple(0.f, 0.0f, 2.0f), 1.0, false, 20, 0, true, false, false);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - Cube 0 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMesh(device, "cube_0", std::make_tuple(0.f, 0.0f, 2.0f), 1.0, false, 20, 0, false, false, false);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - Cube 0 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "cube_0", std::make_tuple(0.f, 0.0f, 2.0f), 1.0, false, 20, 0, false, false, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMesh(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 20, 8, true, false, false);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRasterizeMesh(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 20, 8, false, false, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 4 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 20, 8, true, false, false);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - Bunny Res 4 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_bunny_res4", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 20, 8, false, false, false);
}

TEST_CASE("Test Rasterize Mesh Naive - Bunny Res 2 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_bunny_res2", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 100, 70, true, false, false);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - Bunny Res 2 - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_bunny_res2", std::make_tuple(0.f, -0.1f, 0.3f), 1.0, true, 100, 70, false, false, false);
}

TEST_CASE("Test Rasterize Mesh Naive - 64 Bunnies - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_64_bunny_array", std::make_tuple(0.f, 0.0f, 1.0f), 1.0, true, 25000, 1200, true, false, true);
}

TEST_CASE("Test Rasterize Mesh Coarse-to-Fine - 64 Bunnies - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRasterizeMesh(device, "mesh_64_bunny_array", std::make_tuple(0.f, 0.0f, 1.0f), 1.0, true, 25000, 1200, false, false, true);
}