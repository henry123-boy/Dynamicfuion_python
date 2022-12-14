//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/30/22.
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

#include "tests/test_utils/test_utils.hpp"

#include "geometry/functional/Unproject3dPoints.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestUnproject3dPointsWithoutDepthFiltering(const o3c::Device& device, bool preserve_image_layout) {
	o3c::Tensor intrinsics(std::vector<double>{100., 0., 2.,
	                                           0., 100., 2.,
	                                           0., 0., 1.}, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));
	o3c::Tensor extrinsics = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));
	float depth_scale = 1000.f;
	float depth_max = 3.f;

	const int height = 4;
	const int width = 4;

	o3c::Tensor depth(std::vector<uint16_t>{0, 1000, 2000, 0,
	                                        0, 500, 1500, 0,
	                                        500, 550, 600, 650,
	                                        0, 20, 4000, 0}, {height, width}, o3c::UInt16, device);

	o3c::SizeVector points_gt_size = preserve_image_layout ? o3c::SizeVector{height, width, 3} : o3c::SizeVector{height * width, 3};
	o3c::Tensor points_ground_truth(std::vector<float>{
			0.f, 0.f, 0.f,
			-0.01f, -0.02f, 1.,
			0.f, -0.04f, 2.,
			0.f, 0.f, 0.f,

			0.f, 0.f, 0.f,
			-0.005f, -0.005f, 0.5f,
			0.f, -0.015f, 1.5f,
			0.f, 0.f, 0.f,

			-0.01f, 0.f, 0.5f,
			-0.0055, 0.f, 0.55f,
			0.f, 0.f, 0.6f,
			0.0065, 0.f, 0.65f,

			0.f, 0.f, 0.f,
			-0.0002, 0.0002, 0.02f,
			0.f, 0.f, 0.f, // depth 4m=4000mm is out of default max_depth range
			0.f, 0.f, 0.f
	}, points_gt_size, o3c::Float32, device);

	o3c::SizeVector mask_gt_size = preserve_image_layout ? o3c::SizeVector{height, width} : o3c::SizeVector{height * width};
	o3c::Tensor mask_ground_truth(std::vector<bool>{
			false, true, true, false,
			false, true, true, false,
			true, true, true, true,
			false, true, false, false
	}, mask_gt_size, o3c::Bool, device);

	o3c::Tensor points, mask;
	nnrt::geometry::functional::Unproject3dPointsWithoutDepthFiltering(points, mask, depth, intrinsics, extrinsics, depth_scale, depth_max, preserve_image_layout);

	REQUIRE(points.AllClose(points_ground_truth));
	REQUIRE(mask.AllEqual(mask_ground_truth));
}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering - Preserve Image Layout - CPU") {
	auto device = o3c::Device("CPU:0");
	TestUnproject3dPointsWithoutDepthFiltering(device, true);
}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering  - Preserve Image Layout - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestUnproject3dPointsWithoutDepthFiltering(device, true);
}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering - Don't Preserve Image Layout - CPU") {
	auto device = o3c::Device("CPU:0");
	TestUnproject3dPointsWithoutDepthFiltering(device, false);
}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering  - Don't Preserve Image Layout - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestUnproject3dPointsWithoutDepthFiltering(device, false);
}