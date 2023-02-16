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
// local includes
// test utils
#include "test_utils/test_utils.hpp"
#include "test_main.hpp"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/RasterizeNdcTriangles.h"
#include "core/functional/Masking.h"
#include "alignment/DeformableMeshToImageFitter.h"
// code being tested

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tio = open3d::t::io;
namespace o3tg = open3d::t::geometry;

void TestDeformableImageFitter(const o3c::Device& device, bool draw_depth = false) {
	o3tg::TriangleMesh source_mesh, target_mesh;
	//TODO: add file to test data pack
	o3tio::ReadTriangleMesh(test::generated_mesh_test_data_directory.ToString() + "/plane_skin_source.ply", source_mesh);
	//TODO: add file to test data pack
	o3tio::ReadTriangleMesh(test::generated_mesh_test_data_directory.ToString() + "/plane_skin_target.ply", target_mesh);
	float max_depth = 10.0f;
	float node_coverage = 0.25;

	o3c::SizeVector image_resolution{100, 100};
	o3c::Tensor projection_matrix(
			std::vector<double>{100.0, 0.0, 50.0,
			                    0.0, 100.0, 50.0,
			                    0.0, 0.0, 1.0}, {3, 3},
			o3c::Float64, o3c::Device("CPU:0")
	);

	// flip 180 degrees around the Y-axis, move 1.2 units away from camera
	o3c::Tensor mesh_transform(
			std::vector<double>{-1.0, 0.0, 0.0, 0.0f,
			                   0.0, 1.0, 0.0, 0.0f,
			                   0.0, 0.0, -1.0, 1.2f,
			                   0.0, 0.0, 0.0, 1.0f}, {4, 4},
			o3c::Float64, o3c::Device("CPU:0")
	);

	o3c::Tensor extrinsic_matrix = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));

	// move 1.2 units away from the camera
	target_mesh = target_mesh.Transform(mesh_transform);
	source_mesh = source_mesh.Transform(mesh_transform);

	auto [extracted_face_vertices, clipped_face_mask] =
			nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(target_mesh, projection_matrix, image_resolution, 0.0, max_depth);

	std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
			nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_resolution, 0.f, 1,
			                                       -1, -1, true, false, true);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

	pixel_depths = pixel_depths.Reshape(image_resolution);
	auto zero_bg_pixel_depths = pixel_depths.Clone();
	nnrt::core::functional::ReplaceValue(zero_bg_pixel_depths, -1.0f, 0.0f);
	o3tg::Image depth_image(zero_bg_pixel_depths);
	o3c::Tensor depth_mask =

			(zero_bg_pixel_depths == o3c::Tensor::Zeros(zero_bg_pixel_depths.GetShape(), zero_bg_pixel_depths.GetDtype(), device)).LogicalNot();



	if (draw_depth) {
		auto pd_tmp = pixel_depths.Clone();
		nnrt::core::functional::ReplaceValue(pd_tmp, -1.0f, 10.0f);
		float minimum_depth = pd_tmp.Min({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
		float maximum_depth = pixel_depths.Max({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
		nnrt::core::functional::ReplaceValue(pd_tmp, 10.0f, minimum_depth);
		pd_tmp = 255.f - ((pd_tmp - minimum_depth) * 255.f / (maximum_depth - minimum_depth));
		o3tg::Image stretched_depth_image(pd_tmp.To(o3c::UInt8));
		o3tio::WriteImage(test::generated_image_test_data_directory.ToString() + "/target_depth.png", stretched_depth_image);
	}

	//TODO: add file to test data pack
	o3c::Tensor node_positions = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/nodes.npy");
	//TODO: add file to test data pack
	o3c::Tensor expected_node_translations = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/node_translations.npy");
	//TODO: add file to test data pack
	o3c::Tensor expected_node_rotations = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/node_rotations.npy");
	//TODO: add file to test data pack
	o3c::Tensor edges = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/edges.npy");

	nnrt::geometry::GraphWarpField warp_field(node_positions, edges, o3u::nullopt, o3u::nullopt, node_coverage);

	nnrt::alignment::DeformableMeshToImageFitter fitter(5, 1e-6, true, 10.f);
	o3tg::Image dummy_color_image;


	fitter.FitToImage(warp_field, source_mesh, dummy_color_image, depth_image, depth_mask, projection_matrix, extrinsic_matrix, 1.0f);
	//__DEBUG
	REQUIRE(true);
}

TEST_CASE("Test Deformable Mesh to Image Fitter - CPU") {
	o3c::Device device("CPU:0");
	TestDeformableImageFitter(device);
}

TEST_CASE("Test Deformable Mesh to Image Fitter - CUDA") {
	o3c::Device device("CUDA:0");
	TestDeformableImageFitter(device);
}