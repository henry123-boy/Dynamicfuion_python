//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/19/22.
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
#include "tests/test_utils/test_utils.hpp"

// code being tested
#include "rendering/functional/InterpolateFaceAttributes.h"

namespace o3c = open3d::core;

void TestInterpolateFaceAttributes(const o3c::Device& device, const std::string& mesh_name){
	//FAIL("NOT IMPLEMENTED");
}

TEST_CASE("Test Interpolate Face Attributes - Bunny Res 4 - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInterpolateFaceAttributes(device, "mesh_bunny_res4");
}