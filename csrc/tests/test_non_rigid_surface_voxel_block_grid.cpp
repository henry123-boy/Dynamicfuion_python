//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/9/22.
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
#include <vector>

#include <fmt/ranges.h>
#include <open3d/t/geometry/Image.h>

#include "test_main.hpp"

#include <geometry/NonRigidSurfaceVoxelBlockGrid.h>
#include <geometry/kernel/NonRigidSurfaceVoxelBlockGrid.h>
#include <geometry/GraphWarpField.h>


using namespace nnrt;
namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;


void TestNonRigidSurfaceVoxelBlockGridConstructor(const o3c::Device& device) {
	// simply ensure construction works without bugs
	nnrt::geometry::NonRigidSurfaceVoxelBlockGrid grid(
			std::vector<std::string>({"tsdf", "weight", "color"}),
			std::vector<o3c::Dtype>(
					{o3c::Float32, o3c::UInt16, o3c::UInt16}),
			std::vector<o3c::SizeVector>({{1},
			                              {1},
			                              {3}}),
			0.005f,
			16,
			1000,
			device
	);
}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestNonRigidSurfaceVoxelBlockGridConstructor(device);
}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestNonRigidSurfaceVoxelBlockGridConstructor(device);
}

void TestNonRigidSurfaceVoxelBlockGrid_GetBoundingBoxesOfWarpedBlocks(const o3c::Device& device) {
	std::vector<int32_t> block_key_data{
			0, 0, 0,
			0, 1, 0,
			0, 0, 1,
			0, 1, 1
	};
	o3c::Tensor block_keys{block_key_data, {4, 3}, o3c::Dtype::Int32, device};

	float voxel_size = 0.1;
	int64_t block_resolution = 10;

	std::vector<float> node_data{
			0.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0,
			0.0, 1.0, 1.0,
	};
	o3c::Tensor nodes(node_data, {4, 3}, o3c::Dtype::Float32, device);
	std::vector<int> edge_data{
			1, 2, -1, -1,
			0, 1, -1, -1,
			0, 2, -1, -1,
			1, 2, -1, -1
	};
	o3c::Tensor edges(edge_data, {4, 4}, o3c::Dtype::Int32, device);
	o3c::Tensor edge_weights = o3c::Tensor::Ones({4, 4}, o3c::Dtype::Float32, device);
	o3c::Tensor clusters = o3c::Tensor::Ones({4}, o3c::Dtype::Float32, device);


	geometry::GraphWarpField field(nodes, edges, edge_weights, clusters, 1.0, false, 4);
	// move one unit in the +x direction
	std::vector<float> translation_data{1.0, 0.0, 0.0};
	field.translations.SetItem(o3c::TensorKey::Index(0), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(1), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(2), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(3), o3c::Tensor(translation_data, {3}, o3c::Float32));


	o3c::Tensor bounding_boxes;
	geometry::kernel::voxel_grid::GetBoundingBoxesOfWarpedBlocks(bounding_boxes, block_keys, field, voxel_size, block_resolution,
	                                                             o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")));

	std::vector<float> bounding_boxes_gt_data{
			1, 0, 0, 2, 1, 1,
			1, 1, 0, 2, 2, 1,
			1, 0, 1, 2, 1, 2,
			1, 1, 1, 2, 2, 2
	};
	o3c::Tensor bounding_boxes_gt(bounding_boxes_gt_data, {4, 6}, o3c::Dtype::Float32, device);

	REQUIRE(bounding_boxes.AllClose(bounding_boxes_gt));

}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid GetBoundingBoxesOfWarpedBlocks CPU") {
	auto device = o3c::Device("CPU:0");
	TestNonRigidSurfaceVoxelBlockGrid_GetBoundingBoxesOfWarpedBlocks(device);
}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid GetBoundingBoxesOfWarpedBlocks CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestNonRigidSurfaceVoxelBlockGrid_GetBoundingBoxesOfWarpedBlocks(device);
}


void TestNonRigidSurfaceVoxelBlockGrid_GetAxisAlignedBoxesInterceptingSurfaceMask(const o3c::Device& device) {
	std::vector<float> bounding_boxes_data{
			3, 0, 0, 4, 1, 1,
			3, 1, 0, 4, 2, 1,
			1, 0, 1, 2, 1, 2,
			1, 1, 1, 2, 2, 2
	};
	o3c::Tensor bounding_boxes(bounding_boxes_data, {4, 6}, o3c::Dtype::Float32, device);

	std::vector<float> depth_data{
			1.0, 1.0, 0.0, 0.0,
			1.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.2, 1.2,
			0.0, 0.0, 1.2, 1.2
	};
	o3c::Tensor depth(depth_data, {4, 4}, o3c::Dtype::Float32, device);

	std::vector<double> intrinsics_data{
			1.0, 0.0, 2.0,
			0.0, 1.0, 2.0,
			0.0, 0.0, 1.0
	};
	o3c::Tensor intrinsics(intrinsics_data, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));

	int stride = 1;
	float truncation_distance = 0.5f;

	o3c::Tensor mask;
	geometry::kernel::voxel_grid::GetAxisAlignedBoxesInterceptingSurfaceMask(
			mask, bounding_boxes, intrinsics, depth, 1.0, 100.0, stride, truncation_distance
	);

	o3c::Tensor mask_gt(std::vector<bool>{false, false, true, true}, {4}, o3c::Bool, device);
	REQUIRE(mask.AllEqual(mask_gt));

}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid GetAxisAlignedBoxesInterceptingSurfaceMask CPU") {
	auto device = o3c::Device("CPU:0");
	TestNonRigidSurfaceVoxelBlockGrid_GetAxisAlignedBoxesInterceptingSurfaceMask(device);
}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid GetAxisAlignedBoxesInterceptingSurfaceMask CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestNonRigidSurfaceVoxelBlockGrid_GetAxisAlignedBoxesInterceptingSurfaceMask(device);
}


o3tg::Image GenerateXyPlaneDepthImage(const o3c::SizeVector& size, uint16_t depth, const o3c::Device& device) {
	o3c::Tensor pixels(size, o3c::UInt16, device);
	pixels.Fill(depth);
	return {pixels};
}

o3tg::Image GeneratePlainGrey3ChannelColorImage(const o3c::SizeVector& size, uint8_t value, const o3c::Device& device) {
	o3c::Tensor pixels({size[0], size[1], 3}, o3c::UInt8, device);
	pixels.Fill(value);
	return {pixels};
}

o3c::Tensor ConstructSimpleIntrinsics(double f = 100, double c_x = 50, double c_y = 50) {
	o3c::Device host("CPU:0");
	return o3c::Tensor(
			std::vector<double>({f, 0.0, c_x,
			                     0.0, f, c_y,
			                     0.0, 0.0, 1.0}),
			{3, 3}, o3c::Float64, host
	);
}

void PinchPixelsAtCenter(o3c::Tensor& pixels, int64_t pinch_radius_pixels, float pinch_height_mm, const o3c::Device& device) {
	int64_t image_height = pixels.GetShape(0);
	int64_t image_width = pixels.GetShape(1);
	int64_t pinch_diameter = pinch_radius_pixels * 2;
	// same as np.linspace(-1, 1, pinch_diameter)[None, :] * pinch_height
	o3c::Tensor pinch_y_coordinates = o3c::Tensor::Arange(-1.0, 1.0, 2.0 / static_cast<double>(pinch_diameter - 1), o3c::Float32, device)
			                                  .Append(o3c::Tensor(std::vector<float_t>({1.0f}), {1}, o3c::Dtype::Float32, device)).Reshape(
			{1, pinch_diameter}) * pinch_height_mm;

	// same as np.linspace(-1, 1, pinch_diameter)[:, None] * pinch_height
	o3c::Tensor pinch_x_coordinates = o3c::Tensor::Arange(-1.0, 1.0, 2.0 / static_cast<double>(pinch_diameter - 1), o3c::Float32, device)
			                                  .Append(o3c::Tensor(std::vector<float_t>({1.0f}), {1}, o3c::Dtype::Float32, device)).Reshape(
			{pinch_diameter, 1}) * pinch_height_mm;

	o3c::Tensor y_2 = pinch_y_coordinates * pinch_y_coordinates;
	o3c::Tensor x_2 = pinch_x_coordinates * pinch_x_coordinates;
	o3c::Tensor norms_squared = y_2 + x_2;
	o3c::Tensor delta = -pinch_height_mm + norms_squared.Sqrt();

	int64_t half_image_width = image_width / 2;
	int64_t half_image_height = image_height / 2;
	pixels
		.Slice(0, half_image_height - pinch_radius_pixels, half_image_height + pinch_radius_pixels)
		.Slice(1, half_image_width - pinch_radius_pixels, half_image_width + pinch_radius_pixels) += delta.Round().To(o3c::UInt16);
}

void TestNonRigidSurfaceVoxelBlockGrid_IntegrateNonRigid(const o3c::Device& device) {
	float voxel_size = 0.01f;
	int64_t block_resolution = 8;
	int64_t initial_block_count = 128;
	nnrt::geometry::NonRigidSurfaceVoxelBlockGrid volume(
			std::vector<std::string>({"tsdf", "weight", "color"}),
			std::vector<o3c::Dtype>(
					{o3c::Float32, o3c::UInt16, o3c::UInt16}),
			std::vector<o3c::SizeVector>({{1},
			                              {1},
			                              {3}}),
			voxel_size,
			block_resolution,
			initial_block_count,
			device
	);

	int64_t image_width = 100;
	int64_t image_height = 100;
	o3c::SizeVector image_size({image_width, image_height});
	uint16_t depth_of_plane = 50; // mm
	auto initial_depth_image = GenerateXyPlaneDepthImage(image_size, depth_of_plane, device);
	uint8_t value = 100;
	auto initial_color_image = GeneratePlainGrey3ChannelColorImage(image_size, value, device);

	o3c::Tensor intrinsics = ConstructSimpleIntrinsics();
	o3c::Device host("CPU:0");
	o3c::Tensor extrinsics = o3c::Tensor::Eye(4, o3c::Float64, host);

	float truncation_voxel_multiplier = 2.0f;
	float depth_scale = 1000;
	float depth_max = 3.0;
	auto blocks_to_activate = volume.GetUniqueBlockCoordinates(initial_depth_image, intrinsics, extrinsics, depth_scale, depth_max,
	                                                           truncation_voxel_multiplier);
	volume.Integrate(blocks_to_activate, initial_depth_image, initial_color_image, intrinsics, intrinsics, extrinsics, depth_scale, depth_max,
	                 truncation_voxel_multiplier);

	int64_t node_count = 5;
	o3c::Tensor nodes(std::vector<std::float_t>({0.0, 0.0, 0.05,
	                                             0.02, 0.0, 0.05,
	                                             -0.02, 0.0, 0.05,
	                                             0.00, 0.02, 0.05,
	                                             0.00, -0.02, 0.05}), {node_count, 3}, o3c::Dtype::Float32, device);
	o3c::Tensor node_rotations({node_count, 3, 3}, o3c::Float32, device);
	for (int i_node = 0; i_node < node_count; i_node++) {
		node_rotations.Slice(0, i_node, i_node + 1) = o3c::Tensor::Eye(3, o3c::Float32, device);
	}

	o3c::Tensor node_translations = o3c::Tensor::Zeros({node_count, 3}, o3c::Float32, device);
	node_translations.SetItem(o3c::TensorKey::Index(2), o3c::Tensor(std::vector<float_t>({0.01}), {1}, o3c::Dtype::Float32, device));

	o3c::Tensor deformed_depth_image_pixels(image_size, o3c::UInt16, device);
	deformed_depth_image_pixels.Fill(depth_of_plane);

	PinchPixelsAtCenter(deformed_depth_image_pixels, 20, 10.0f, device);



}


TEST_CASE("Test Non-Rigid Surface Voxel Block Grid IntegrateNonRigid CPU") {
	auto device = o3c::Device("CPU:0");
	TestNonRigidSurfaceVoxelBlockGrid_IntegrateNonRigid(device);
}

TEST_CASE("Test Non-Rigid Surface Voxel Block Grid IntegrateNonRigid CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestNonRigidSurfaceVoxelBlockGrid_IntegrateNonRigid(device);
}