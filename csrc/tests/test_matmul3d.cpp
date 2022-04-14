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

#include <open3d/core/Tensor.h>
#include "core/TensorManipulationRoutines.h"

#include <Eigen/Dense>


using namespace nnrt;
namespace o3c = open3d::core;


void TestMatmul3D(const o3c::Device& device) {
	int64_t batch_size = 3;
	// Note: inner matrices are column-major, so...
	// Be sure to run moveaxis in numpy before copy/pasting the values for testing:
	// np.moveaxis(A, (0,1,2),(0,2,1)).flatten()
	// std::vector<float> A_data = {7., 5., 2., 4., 1., 4., 2., 2., 9., 8., 7., 7., 1., 9., 1., 6., 2., 0.};
	// o3c::Tensor A(A_data, {3, 2, 3}, o3c::Dtype::Float32, device);
	// std::vector<float> B_data = {9.,  6.,  5.,  4.,  9., 10.,  2.,  3.,  5.,  6.,  6.,  1.,  6., 7.,  2.,  4.,  4.,  5.};
	// o3c::Tensor B(B_data, {3, 3, 2}, o3c::Dtype::Float32, device);
	//
	// auto C = core::Matmul3D(A,B);
	//
	// auto C_data = C.ToFlatVector<float>();
	//
	// std::vector<float> C_gt_data = {80., 89., 56., 96., 66., 63., 73., 67., 17., 96., 18., 60.};
	std::vector<float> A_data = {7., 2., 1., 5., 4., 4., 2., 9., 7., 2., 8., 7., 1., 1., 2., 9., 6.,0.};
	o3c::Tensor A(A_data, {3, 2, 3}, o3c::Dtype::Float32, device);
	std::vector<float> B_data = {9.,  4.,  6.,  9.,  5., 10.,  2.,  6.,  3.,  6.,  5.,  1.,  6., 4.,  7.,  4.,  2.,  5.};
	o3c::Tensor B(B_data, {3, 3, 2}, o3c::Dtype::Float32, device);

	auto C = core::Matmul3D(A,B);

	auto C_data = C.ToFlatVector<float>();

	std::vector<float> C_gt_data = {80., 56., 89., 96., 66., 73., 63., 67., 17., 18., 96., 60.};


	REQUIRE(std::equal(C_data.begin(), C_data.end(), C_gt_data.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-6).epsilon(1e-12); }));


}

TEST_CASE("Test Matmul3D CPU") {
	auto device = o3c::Device("CPU:0");
	TestMatmul3D(device);
}

TEST_CASE("Test Matmul3D CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestMatmul3D(device);
}