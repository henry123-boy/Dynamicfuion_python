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

#include "geometry/Unproject3dPoints.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestUnproject3dPointsWithoutDepthFiltering(const o3c::Device& device) {
	o3c::Tensor intrinsics(std::vector<float> {100.f, 0.f, 2.f,
											   0.f, 100.f, 2.f,
											   0.f, 0.f, 1.f}, {3,3}, o3c::Float32, o3c::Device("CPU:0") );

}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering CPU") {
	auto device = o3c::Device("CPU:0");
	TestUnproject3dPointsWithoutDepthFiltering(device);
}

TEST_CASE("Test Unproject 3D Points Without Depth Filtering CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestUnproject3dPointsWithoutDepthFiltering(device);
}