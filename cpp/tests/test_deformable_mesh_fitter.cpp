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
#include <Eigen/Dense>
// local includes
// test utils
#include "test_main.hpp"
// code being tested

namespace o3c = open3d::core;

TEST_CASE("experiment") {
	// auto device = o3c::Device("CUDA:0");
	// auto tensor = o3c::Tensor::Zeros({4}, o3c::Float32, device);
	// auto mask = o3c::Tensor(std::vector<bool>({true, false, true, false}), {4}, o3c::Bool, device);
	// auto value = o3c::Tensor::Ones({1}, o3c::Float32, device);
	// tensor.SetItem(o3c::TensorKey::IndexTensor(mask), value);
	// auto gt = o3c::Tensor(std::vector<float>({1.0, 0.0, 1.0, 0.0}), {4}, o3c::Float32, device);
	// std::cout << tensor.ToString() << std::endl;
	// std::cout << gt.ToString() << std::endl;
	// REQUIRE(tensor.AllClose(gt));
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> x;
	Eigen::Matrix<float, 3, 3, Eigen::ColMajor> y;
	Eigen::Matrix<float, 3, 9, Eigen::RowMajor> z = Eigen::Matrix<float, 3, 9, Eigen::RowMajor>::Random();


	Eigen::Vector3f a(1.f, 2.f, 3.f);
	Eigen::Vector3f b(4.f, 5.f, 6.f);
	Eigen::Vector3f c(7.f, 8.f, 9.f);


	int repetition_count = 1000000;

	auto begin = std::chrono::high_resolution_clock::now();
	for(int i = 0; i< repetition_count; i++){
		x << a, b, c;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "Total time elapsed for all x (row-major) << a, b, c; repetitions: " << elapsed.count() << " ms" << std::endl;

	begin = std::chrono::high_resolution_clock::now();
	for(int i = 0; i< repetition_count; i++){
		y << a, b, c;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "Total time elapsed for all y (column-major) << a, b, c; repetitions: " << elapsed.count() << " ms" << std::endl;

	begin = std::chrono::high_resolution_clock::now();
	for(int i = 0; i< repetition_count; i++){
		auto e = x * z;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "Total time elapsed for all e = x * z; repetitions: " << elapsed.count() << " ms" << std::endl;

	begin = std::chrono::high_resolution_clock::now();
	for(int i = 0; i< repetition_count; i++){
		auto e = y * z;
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << "Total time elapsed for all e = y * z; repetitions: " << elapsed.count() << " ms" << std::endl;

}