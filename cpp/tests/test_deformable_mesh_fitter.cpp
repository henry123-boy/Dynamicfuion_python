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
// local includes
// test utils
#include "test_main.hpp"
// code being tested

namespace o3c = open3d::core;

TEST_CASE("experiment") {
	auto device = o3c::Device("CUDA:0");
	auto tensor = o3c::Tensor::Zeros({4}, o3c::Float32, device);
	auto mask = o3c::Tensor(std::vector<bool>({true, false, true, false}), {4}, o3c::Bool, device);
	auto value = o3c::Tensor::Ones({1}, o3c::Float32, device);
	tensor.SetItem(o3c::TensorKey::IndexTensor(mask), value);
	auto gt = o3c::Tensor(std::vector<float>({1.0, 0.0, 1.0, 0.0}), {4}, o3c::Float32, device);
	std::cout << tensor.ToString() << std::endl;
	std::cout << gt.ToString() << std::endl;
	REQUIRE(tensor.AllClose(gt));
}