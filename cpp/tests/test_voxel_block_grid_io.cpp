//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/11/22.
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
#include "test_main.hpp"

// 3rd party
#include <open3d/core/Device.h>
#include <open3d/geometry/Image.h>
#include <open3d/io/ImageIO.h>
#include <open3d/t/geometry/Image.h>

// nnrt
#include "geometry/VoxelBlockGrid.h"
#include "geometry/NonRigidSurfaceVoxelBlockGrid.h"
#include "io/VoxelBlockGridIO.h"
#include "tests/test_utils/test_utils.hpp"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

template<typename TVoxelBlockGrid>
void TestVoxelBlockGrid_SaveLoad(const o3c::Device& device) {
	TVoxelBlockGrid saved_grid(
			std::vector<std::string>({"tsdf", "weight", "color"}),
			std::vector<o3c::Dtype>({o3c::Float32, o3c::UInt16, o3c::UInt16}),
			std::vector<o3c::SizeVector>({{1}, {1}, {3}}),
			0.005, 16, 1000, device
	);

	o3g::Image depth_legacy;
	std::string depth_path = test::static_image_test_data_directory.ToString() + "/red_shorts_200_depth.png";
	open3d::io::ReadImage(depth_path, depth_legacy);
	o3tg::Image depth = o3tg::Image::FromLegacy(depth_legacy, device);

	o3g::Image color_legacy;
	std::string color_path = test::static_image_test_data_directory.ToString() + "/red_shorts_200_color.jpg";
	open3d::io::ReadImage(color_path, color_legacy);
	o3tg::Image color = o3tg::Image::FromLegacy(color_legacy, device);

	std::string intrinsics_path = test::static_intrinsics_test_data_directory.ToString() + "/red_shorts_intrinsics.txt";
	o3c::Device host("CPU:0");
	o3c::Tensor intrinsics = o3c::Tensor(test::read_intrinsics(test::static_intrinsics_test_data_directory.ToString() + "/red_shorts_intrinsics.txt"),
	                                     {4, 4}, o3c::Float64, host).Slice(0, 0, 3).Slice(1, 0, 3).Contiguous();
	o3c::Tensor extrinsics = o3c::Tensor::Eye(4, o3c::Float64, host);

	float depth_scale = 1000.0f;
	float depth_max = 3.0f;
	float truncation_voxel_multiplier = 5.0f;
	o3c::Tensor block_coordinates = saved_grid.GetUniqueBlockCoordinates(depth, intrinsics, extrinsics, depth_scale, depth_max,
	                                                                     truncation_voxel_multiplier);
	saved_grid.Integrate(block_coordinates, depth, color, intrinsics, extrinsics, depth_scale, depth_max, truncation_voxel_multiplier);
	std::string output_path = test::generated_test_data_directory.ToString() + "test_voxel_block_grid_save_load.dat";
	nnrt::io::WriteVoxelBlockGrid(output_path, saved_grid);

	nnrt::geometry::VoxelBlockGrid loaded_grid_host;
	nnrt::io::ReadVoxelBlockGrid(output_path, loaded_grid_host);
	nnrt::geometry::VoxelBlockGrid loaded_grid = loaded_grid_host.To(device);

	REQUIRE(loaded_grid == saved_grid);
}

TEST_CASE("Test VoxelBlockGrid Save Load CPU") {
	o3c::Device device("CPU:0");
	TestVoxelBlockGrid_SaveLoad<nnrt::geometry::VoxelBlockGrid>(device);
}

TEST_CASE("Test VoxelBlockGrid Save Load CUDA") {
	o3c::Device device("CUDA:0");
	TestVoxelBlockGrid_SaveLoad<nnrt::geometry::VoxelBlockGrid>(device);
}

TEST_CASE("Test NonRigidSurfaceVoxelBlockGrid Save Load CPU") {
	o3c::Device device("CPU:0");
	TestVoxelBlockGrid_SaveLoad<nnrt::geometry::NonRigidSurfaceVoxelBlockGrid>(device);
}

TEST_CASE("Test NonRigidSurfaceVoxelBlockGrid Save Load CUDA") {
	o3c::Device device("CUDA:0");
	TestVoxelBlockGrid_SaveLoad<nnrt::geometry::NonRigidSurfaceVoxelBlockGrid>(device);
}
