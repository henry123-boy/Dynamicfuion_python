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

#include <open3d/core/Tensor.h>
#include "io/TensorIO.h"
#include "tests/test_utils/test_utils.hpp"

namespace o3c = open3d::core;

void TestTensorSaveLoad(const o3c::Device& device) {

	o3c::Tensor saved_tensor(
			std::vector<float>(
					{0.09923797f, 0.42924188f, 0.66608149f, 0.48765226f, 0.0133136f,
					 0.53882037f, 0.12795316f, 0.71762794f, 0.11413502f, 0.07152748f,
					 0.87123131f, 0.42435338f, 0.24144447f, 0.61602327f, 0.13903003f,
					 0.04249958f, 0.18954649f, 0.61328016f, 0.85281063f, 0.65742497f}
			), {5, 2, 2}, o3c::Float32, device);
	std::string output_path = test::generated_test_data_directory.ToString() + "test_tensor_save_load.dat";

	nnrt::io::WriteTensor(output_path, saved_tensor);
	auto loaded_tensor = nnrt::io::ReadTensor(output_path);

	o3c::Device host("CPU:0");
	REQUIRE(loaded_tensor.GetDevice() == host);
	auto loaded_tensor_device = loaded_tensor.To(device);
	REQUIRE(saved_tensor.AllEqual(loaded_tensor_device));
}

TEST_CASE("Test Tensor Save Load CPU") {
	o3c::Device device("CPU:0");
	TestTensorSaveLoad(device);
}

TEST_CASE("Test Tensor Save Load CUDA") {
	o3c::Device device("CUDA:0");
	TestTensorSaveLoad(device);
}

void TestTensorSliceSaveLoad(const o3c::Device& device) {
	using namespace nnrt::io;
	o3c::Tensor saved_tensor(
			std::vector<float>(
					{0.09923797f, 0.42924188f, 0.66608149f, 0.48765226f, 0.0133136f,
					 0.53882037f, 0.12795316f, 0.71762794f, 0.11413502f, 0.07152748f,
					 0.87123131f, 0.42435338f, 0.24144447f, 0.61602327f, 0.13903003f,
					 0.04249958f, 0.18954649f, 0.61328016f, 0.85281063f, 0.65742497f}
			), {5, 2, 2}, o3c::Float32, device);
	std::string output_path = test::generated_test_data_directory.ToString() + "/test_tensor_slice_save_load.dat";

	nnrt::io::WriteTensor(output_path, saved_tensor.Slice(0, 1, 4, 2));
	auto loaded_tensor = nnrt::io::ReadTensor(output_path);

	o3c::Device host("CPU:0");
	REQUIRE(loaded_tensor.GetDevice() == host);
	auto loaded_tensor_device = loaded_tensor.To(device);
	REQUIRE(saved_tensor.Slice(0, 1, 4, 2).AllEqual(loaded_tensor_device));
}

TEST_CASE("Test Tensor Slice Save Load CPU") {
	o3c::Device device("CPU:0");
	TestTensorSaveLoad(device);
}

TEST_CASE("Test Tensor Slice Save Load CUDA") {
	o3c::Device device("CUDA:0");
	TestTensorSaveLoad(device);
}