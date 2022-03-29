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
#include <fmt/ranges.h>

#include "test_main.hpp"

#include <geometry/WarpableTSDFVoxelGrid.h>


using namespace nnrt;
namespace o3c = open3d::core;


void TestWarpableTSDFVoxelGrid(const o3c::Device& device) {
	// simply ensure construction works without bugs
	nnrt::geometry::WarpableTSDFVoxelGrid grid(
			{
					{"tsdf",   o3c::Float32},
					{"weight", o3c::UInt16},
					{"color",  o3c::UInt16}
			},
			0.005f,
			0.025f,
			16,
			1000,
			device
	);
}

TEST_CASE("Test Warpable TSDF Voxel Grid CPU") {
	auto device = o3c::Device("CPU:0");
	TestWarpableTSDFVoxelGrid(device);
}

TEST_CASE("Test Warpable TSDF Voxel Grid CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestWarpableTSDFVoxelGrid(device);
}