//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/1/21.
//  Copyright (c) 2021 Gregory Kramida
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

#include <core/KdTree.h>
#include <open3d/core/Tensor.h>

using namespace nnrt;
namespace o3c = open3d::core;

TEST_CASE("Test 1D KDTree Construction CPU") {
	auto device = o3c::Device("cpu:0");
	std::vector<float> kd_tree_point_data{5.f, 2.f, 1.f, 6.0f, 3.f, 4.f};
	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 1}, o3c::Dtype::Float32, device);

	core::KdTree kd_tree(kd_tree_points);

}