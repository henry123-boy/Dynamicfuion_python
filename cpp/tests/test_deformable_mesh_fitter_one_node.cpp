//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
#include <open3d/t/io/TriangleMeshIO.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/geometry/Geometry.h>

#include <utility>
// local includes
// test utils
#include "test_utils/test_utils.hpp"
#include "test_main.hpp"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/RasterizeNdcTriangles.h"
#include "core/functional/Masking.h"
#include "alignment/DeformableMeshToImageFitter.h"
#include "tests/test_utils/fitter_testing.h"
// code being tested

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tio = open3d::t::io;
namespace o3tg = open3d::t::geometry;

void DrawDepth(const o3c::Tensor& pixel_depths, const std::string& image_name) {
	auto pd_tmp = pixel_depths.Clone();
	nnrt::core::functional::ReplaceValue(pd_tmp, -1.0f, 10.0f);
	float minimum_depth = pd_tmp.Min({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
	float maximum_depth = pixel_depths.Max({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
	nnrt::core::functional::ReplaceValue(pd_tmp, 10.0f, minimum_depth);
	pd_tmp = 255.f - ((pd_tmp - minimum_depth) * 255.f / (maximum_depth - minimum_depth));
	o3tg::Image stretched_depth_image(pd_tmp.To(o3c::UInt8));
	o3tio::WriteImage(test::generated_image_test_data_directory.ToString() + "/" + image_name + ".png",
	                  stretched_depth_image);
}

void TestDeformableImageFitter_1NodePlaneTranslation(
		const o3c::Device& device, bool use_perspective_correction = false,
		std::vector<nnrt::alignment::IterationMode> iteration_modes = {nnrt::alignment::IterationMode::ALL},
		int max_iterations = 1,
		nnrt::geometry::WarpNodeCoverageComputationMethod warp_node_coverage_computation_method =
		nnrt::geometry::WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE,
		bool draw_depth = false
) {
	float max_depth = 10.0f;
	float node_coverage = 0.25;

	// flip 180 degrees around the Y-axis, move 1.2 units away from camera
	o3c::Tensor mesh_transform(
			std::vector<double>{-1.0, 0.0, 0.0, 0.0f,
			                    0.0, 1.0, 0.0, 0.0f,
			                    0.0, 0.0, -1.0, 1.2f,
			                    0.0, 0.0, 0.0, 1.0f}, {4, 4},
			o3c::Float64, o3c::Device("CPU:0")
	);

	//TODO: add files to test data pack
	auto [source_mesh, target_mesh] =
			test::ReadAndTransformTwoMeshes("plane_skin_1_node_source", "plane_skin_1_node_translation_target", device,
			                                mesh_transform);


	o3c::SizeVector image_resolution{100, 100};
	o3c::Tensor projection_matrix(
			std::vector<double>{100.0, 0.0, 50.0,
			                    0.0, 100.0, 50.0,
			                    0.0, 0.0, 1.0}, {3, 3},
			o3c::Float64, o3c::Device("CPU:0")
	);

	auto [extracted_face_vertices, clipped_face_mask] =
			nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(target_mesh, projection_matrix,
			                                                               image_resolution, 0.0, max_depth);

	std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
			nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_resolution, 0.5f, 1, -1, -1,
			                                       use_perspective_correction, false, true);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

	pixel_depths = pixel_depths.Reshape(image_resolution);
	auto zero_bg_pixel_depths = pixel_depths.Clone();
	nnrt::core::functional::ReplaceValue(zero_bg_pixel_depths, -1.0f, 0.0f);
	o3tg::Image depth_image(zero_bg_pixel_depths);
	o3c::Tensor depth_mask =
			(zero_bg_pixel_depths ==
			 o3c::Tensor::Zeros(zero_bg_pixel_depths.GetShape(), zero_bg_pixel_depths.GetDtype(), device)).LogicalNot();

	if (draw_depth) {
		DrawDepth(pixel_depths, "target_depth_1-node_plane_translation");
	}

	o3c::Tensor node_positions = o3c::Tensor(std::vector<float>{0.0, 0.0, 1.2}, {1, 3}, o3c::Float32, device);


	o3c::Tensor expected_node_translations = o3c::Tensor(std::vector<float>{0.0, 0.0, 0.2}, {1, 3}, o3c::Float32, device);
	o3c::Tensor expected_node_rotations = o3c::Tensor(std::vector<float>{1.0, 0.0, 0.0,
	                                                                     0.0, 1.0, 0.0,
	                                                                     0.0, 0.0, 1.0}, {1, 3, 3}, o3c::Float32, device);


	nnrt::geometry::HierarchicalGraphWarpField warp_field(node_positions, node_coverage, false, 1, 0, warp_node_coverage_computation_method, 1);

	nnrt::alignment::DeformableMeshToImageFitter fitter(max_iterations, std::move(iteration_modes), 1e-6,
	                                                    use_perspective_correction, 10.f, false, 0.01, 0.001);
	o3tg::Image dummy_color_image;

	o3c::Tensor extrinsic_matrix = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));
	fitter.FitToImage(warp_field, source_mesh, dummy_color_image, depth_image, depth_mask, projection_matrix,
	                  extrinsic_matrix, 1.0f);

	REQUIRE(warp_field.GetNodeRotations().AllClose(expected_node_rotations, 1., 1e-5));
	REQUIRE(warp_field.GetNodeTranslations().AllClose(expected_node_translations, 1., 1e-5));
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane Translation - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneTranslation(
			device, true, {nnrt::alignment::IterationMode::ALL}, 3,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - COMBINED - MIN - 1 Node Plane Translation - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneTranslation(
			device, true, {nnrt::alignment::IterationMode::ALL}, 3,
			nnrt::geometry::WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE
	);
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane Translation - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlaneTranslation(
			device, true, {nnrt::alignment::IterationMode::ALL}, 3,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - TRANSLATION-ONLY - 1 Node Plane Translation - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneTranslation(
			device, true, {nnrt::alignment::IterationMode::TRANSLATION_ONLY}, 3,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - TRANSLATION-ONLY - 1 Node Plane Translation - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlaneTranslation(
			device, true, {nnrt::alignment::IterationMode::TRANSLATION_ONLY}, 3,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}


void TestDeformableImageFitter_1NodePlaneRotation(
		const o3c::Device& device, int angle,
		bool use_perspective_correction = false,
		std::vector<nnrt::alignment::IterationMode> iteration_modes = {
				nnrt::alignment::IterationMode::ALL},
		int max_iterations = 1,
		nnrt::geometry::WarpNodeCoverageComputationMethod warp_node_coverage_computation_method =
		nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE,
		bool draw_depth = true
) {
	float max_depth = 10.0f;
	float node_coverage = 0.25;

	// flip 180 degrees around the Y-axis, move 1.2 units away from camera
	o3c::Tensor mesh_transform(
			std::vector<double>{-1.0, 0.0, 0.0, 0.0f,
			                    0.0, 1.0, 0.0, 0.0f,
			                    0.0, 0.0, -1.0, 1.2f,
			                    0.0, 0.0, 0.0, 1.0f}, {4, 4},
			o3c::Float64, o3c::Device("CPU:0")
	);

	//TODO: add files to test data pack
	auto [source_mesh, target_mesh] =
			test::ReadAndTransformTwoMeshes("plane_skin_1_node_source", "plane_skin_1_node_rotation_"
			                                                            + std::to_string(angle) + "_target", device, mesh_transform);


	o3c::SizeVector image_resolution{100, 100};
	o3c::Tensor projection_matrix(
			std::vector<double>{100.0, 0.0, 50.0,
			                    0.0, 100.0, 50.0,
			                    0.0, 0.0, 1.0}, {3, 3},
			o3c::Float64, o3c::Device("CPU:0")
	);

	auto [extracted_face_vertices, clipped_face_mask] =
			nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(target_mesh, projection_matrix,
			                                                               image_resolution, 0.0, max_depth);

	std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
			nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_resolution, 0.f, 1,
			                                       -1, -1, use_perspective_correction, false, true);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

	pixel_depths = pixel_depths.Reshape(image_resolution);
	auto zero_bg_pixel_depths = pixel_depths.Clone();
	nnrt::core::functional::ReplaceValue(zero_bg_pixel_depths, -1.0f, 0.0f);
	o3tg::Image depth_image(zero_bg_pixel_depths);
	o3c::Tensor depth_mask =
			(zero_bg_pixel_depths ==
			 o3c::Tensor::Zeros(zero_bg_pixel_depths.GetShape(), zero_bg_pixel_depths.GetDtype(), device)).LogicalNot();

	if (draw_depth) {
		DrawDepth(pixel_depths, "target_depth_1-node_plane_rotation_" + std::to_string(angle));
	}

	o3c::Tensor node_positions = o3c::Tensor(std::vector<float>{0.0, 0.0, 1.2}, {1, 3}, o3c::Float32, device);

	o3c::Tensor expected_node_translations = o3c::Tensor(std::vector<float>{0.0, 0.0, 0.0}, {1, 3}, o3c::Float32,
	                                                     device);

	nnrt::geometry::HierarchicalGraphWarpField warp_field(node_positions, node_coverage, false, 1, 0, warp_node_coverage_computation_method, 1);

	nnrt::alignment::DeformableMeshToImageFitter fitter(
			max_iterations, std::move(iteration_modes), 1e-6,
			use_perspective_correction, 10.f, false, 0.01f, 0.001f
	);
	o3tg::Image dummy_color_image;

	o3c::Tensor extrinsic_matrix = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));
	fitter.FitToImage(warp_field, source_mesh, dummy_color_image, depth_image, depth_mask, projection_matrix,
	                  extrinsic_matrix, 1.0f);

	// rotation about x, flip sign to account for world-to-camera transform
	Eigen::Vector3d rotation_xyz(static_cast<double>(-angle * M_PI / 180.0), 0.0, 0.0);
	Eigen::Matrix3f rotation_eigen = open3d::geometry::Geometry3D::GetRotationMatrixFromXYZ(
			rotation_xyz).cast<float>();

	o3c::Tensor expected_node_rotations =
			o3c::Tensor(
					std::vector<float>{
							rotation_eigen(0, 0), rotation_eigen(0, 1), rotation_eigen(0, 2),
							rotation_eigen(1, 0), rotation_eigen(1, 1), rotation_eigen(1, 2),
							rotation_eigen(2, 0), rotation_eigen(2, 1), rotation_eigen(2, 2)},
					{1, 3, 3}, o3c::Float32, device
			);

	REQUIRE(warp_field.GetNodeRotations().AllClose(expected_node_rotations, 1., 1e-4));
	REQUIRE(warp_field.GetNodeTranslations().AllClose(expected_node_translations, 1., 1e-3));
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane Rotation x 45 - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 45, true, {nnrt::alignment::IterationMode::ALL}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - COMBINED - MIN - 1 Node Plane Rotation x 45 - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 45, true, {nnrt::alignment::IterationMode::ALL}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE
	);
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane Rotation x 45 - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 45, true, {nnrt::alignment::IterationMode::ALL}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}


TEST_CASE("Test DMI Fitter - ROTATION-ONLY - 1 Node Plane Rotation x 45 - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 45, true, {nnrt::alignment::IterationMode::ROTATION_ONLY}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - ROTATION-ONLY - 1 Node Plane Rotation x 45 - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 45, true, {nnrt::alignment::IterationMode::ROTATION_ONLY}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - ROTATION-ONLY - 1 Node Plane Rotation x 5 - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, 5, true, {nnrt::alignment::IterationMode::ROTATION_ONLY}, 2,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - ROTATION-ONLY - 1 Node Plane Rotation x -45 - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, -45, true, {nnrt::alignment::IterationMode::ROTATION_ONLY}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - ROTATION-ONLY - 1 Node Plane Rotation x -45 - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlaneRotation(
			device, -45, true, {nnrt::alignment::IterationMode::ROTATION_ONLY}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}


void TestDeformableImageFitter_1NodePlane(
		const o3c::Device& device,
		bool use_perspective_correction = true,
		std::vector<nnrt::alignment::IterationMode> iteration_modes = {
				nnrt::alignment::IterationMode::ALL},
		int max_iterations = 1,
		nnrt::geometry::WarpNodeCoverageComputationMethod warp_node_coverage_computation_method =
		nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE,
		bool draw_depth = true
) {
	float max_depth = 10.0f;
	float node_coverage = 0.25;

	// flip 180 degrees around the Y-axis, move 1.2 units away from camera
	o3c::Tensor mesh_transform(
			std::vector<double>{-1.0, 0.0, 0.0, 0.0f,
			                    0.0, 1.0, 0.0, 0.0f,
			                    0.0, 0.0, -1.0, 1.2f,
			                    0.0, 0.0, 0.0, 1.0f}, {4, 4},
			o3c::Float64, o3c::Device("CPU:0")
	);

	//TODO: add files to test data pack
	auto [source_mesh, target_mesh] =
			test::ReadAndTransformTwoMeshes("plane_skin_1_node_source", "plane_skin_1_node_target", device, mesh_transform);


	o3c::SizeVector image_resolution{100, 100};
	o3c::Tensor projection_matrix(
			std::vector<double>{100.0, 0.0, 50.0,
			                    0.0, 100.0, 50.0,
			                    0.0, 0.0, 1.0}, {3, 3},
			o3c::Float64, o3c::Device("CPU:0")
	);

	auto [extracted_face_vertices, clipped_face_mask] =
			nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(target_mesh, projection_matrix,
			                                                               image_resolution, 0.0, max_depth);

	std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
			nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_resolution, 0.f, 1,
			                                       -1, -1, use_perspective_correction, false, true);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

	pixel_depths = pixel_depths.Reshape(image_resolution);
	auto zero_bg_pixel_depths = pixel_depths.Clone();
	nnrt::core::functional::ReplaceValue(zero_bg_pixel_depths, -1.0f, 0.0f);
	o3tg::Image depth_image(zero_bg_pixel_depths);
	o3c::Tensor depth_mask =
			(zero_bg_pixel_depths ==
			 o3c::Tensor::Zeros(zero_bg_pixel_depths.GetShape(), zero_bg_pixel_depths.GetDtype(), device)).LogicalNot();

	if (draw_depth) {
		DrawDepth(pixel_depths, "target_depth_1-node_plane");
	}

	o3c::Tensor node_positions = o3c::Tensor(std::vector<float>{0.0, 0.0, 1.2}, {1, 3}, o3c::Float32, device);

	o3c::Tensor expected_node_translations = o3c::Tensor(std::vector<float>{0.0, 0.0, 0.2}, {1, 3}, o3c::Float32,
	                                                     device);

	o3c::Tensor edges = o3c::Tensor(std::vector<int>{-1, -1, -1, -1}, {1, 4}, o3c::Int32, device);

	nnrt::geometry::HierarchicalGraphWarpField warp_field(node_positions, node_coverage, false, 1, 0, warp_node_coverage_computation_method, 1);

	nnrt::alignment::DeformableMeshToImageFitter fitter(
			max_iterations, std::move(iteration_modes), 1e-6,
			use_perspective_correction, 10.f, false, 0.01f, 0.001f
	);
	o3tg::Image dummy_color_image;

	o3c::Tensor extrinsic_matrix = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));
	fitter.FitToImage(warp_field, source_mesh, dummy_color_image, depth_image, depth_mask, projection_matrix,
	                  extrinsic_matrix, 1.0f);

	// rotation about x, flip sign to account for world-to-camera transform
	float angle = 45.f;
	Eigen::Vector3d rotation_xyz(static_cast<double>(-angle * M_PI / 180.0), 0.0, 0.0);
	Eigen::Matrix3f rotation_eigen = open3d::geometry::Geometry3D::GetRotationMatrixFromXYZ(
			rotation_xyz).cast<float>();

	o3c::Tensor expected_node_rotations =
			o3c::Tensor(
					std::vector<float>{
							rotation_eigen(0, 0), rotation_eigen(0, 1), rotation_eigen(0, 2),
							rotation_eigen(1, 0), rotation_eigen(1, 1), rotation_eigen(1, 2),
							rotation_eigen(2, 0), rotation_eigen(2, 1), rotation_eigen(2, 2)},
					{1, 3, 3}, o3c::Float32, device
			);

	REQUIRE(warp_field.GetNodeRotations().AllClose(expected_node_rotations, 1., 1e-3));
	REQUIRE(warp_field.GetNodeTranslations().AllClose(expected_node_translations, 1., 1e-3));
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter_1NodePlane(
			device, true, {nnrt::alignment::IterationMode::ALL}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}

TEST_CASE("Test DMI Fitter - COMBINED - FIXED - 1 Node Plane - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter_1NodePlane(
			device, true, {nnrt::alignment::IterationMode::ALL}, 4,
			nnrt::geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE
	);
}