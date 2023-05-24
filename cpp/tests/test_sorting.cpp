//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/24/23.
//  Copyright (c) 2023 Gregory Kramida
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
#include "test_main.hpp"
// code being tested
#include "core/functional/SortOrder.h"
#include "core/functional/Sorting.h"

namespace o3c = open3d::core;

void TestSortingAlongLastDimension(const o3c::Device& device) {
	o3c::Tensor tensor(std::vector<int>{
			1, 3, 4, 2, 5,
			4, -1, -2, 3, -1,
			0, -1, -4, 2, 1
	}, {3, 5}, o3c::Int32, device);

	auto sorted_asc = nnrt::core::functional::SortTensorAlongLastDimension(tensor, false, nnrt::core::functional::SortOrder::ASC);
	o3c::Tensor tensor_gt_asc(
			std::vector<int>{
					1, 2, 3, 4, 5,
					-2, -1, -1, 3, 4,
					-4, -1, 0, 1, 2
			}, {3, 5}, o3c::Int32, device
	);
	REQUIRE(sorted_asc.AllEqual(tensor_gt_asc));

	auto sorted_asc_nnf = nnrt::core::functional::SortTensorAlongLastDimension(tensor, true, nnrt::core::functional::SortOrder::ASC);
	o3c::Tensor tensor_gt_asc_nnf(
			std::vector<int>{
					1, 2, 3, 4, 5,
					3, 4, -2, -1, -1,
					0, 1, 2, -4, -1
			}, {3, 5}, o3c::Int32, device
	);

	REQUIRE(sorted_asc_nnf.AllEqual(tensor_gt_asc_nnf));

	auto sorted_desc = nnrt::core::functional::SortTensorAlongLastDimension(tensor, true, nnrt::core::functional::SortOrder::DESC);
	o3c::Tensor tensor_gt_desc(
			std::vector<int>{
					5, 4, 3, 2, 1,
					4, 3, -1, -1, -2,
					2, 1, 0, -1, -4
			}, {3, 5}, o3c::Int32, device
	);

	REQUIRE(sorted_desc.AllEqual(tensor_gt_desc));

	auto sorted_desc_pf = nnrt::core::functional::SortTensorAlongLastDimension(tensor, false, nnrt::core::functional::SortOrder::DESC);
	o3c::Tensor tensor_gt_desc_pf(
			std::vector<int>{
					5, 4, 3, 2, 1,
					-1, -1, -2, 4, 3,
					-1, -4, 2, 1, 0
			}, {3, 5}, o3c::Int32, device
	);
	REQUIRE(sorted_desc_pf.AllEqual(tensor_gt_desc_pf));
}


TEST_CASE("Test Sorting Along Last Dimension - CPU") {
	auto device = o3c::Device("CPU:0");
	TestSortingAlongLastDimension(device);
}

TEST_CASE("Test Sorting Along Last Dimension - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestSortingAlongLastDimension(device);
}


void TestSortingAlongLastDimensionByKey(const o3c::Device& device) {
	o3c::Tensor key_tensor(std::vector<float>{
			1.f, 3.f, 4.f, 2.f, 5.f,    // 0, 1, 2, 3, 4
			4.f, -1.f, -2.f, 3.f, -1.f, // 5, 6, 7, 8, 9,
			0.f, -1.f, -4.f, 2.f, 1.f   // 10, 11, 12, 13, 14
	}, {3, 5}, o3c::Float32, device);

	o3c::Tensor value_tensor(std::vector<int>{
			0, 1, 2, 3, 4,      // 1,  3,  4, 2,  5,
			5, 6, 7, 8, 9,      // 4, -1, -2, 3, -1,
			10, 11, 12, 13, 14  // 0, -1, -4, 2,  1
	}, {3, 5}, o3c::Int32, device);

	o3c::Tensor values_asc, keys_asc;
	std::tie(values_asc, keys_asc) = nnrt::core::functional::SortTensorAlongLastDimensionByKey(value_tensor, key_tensor, false,
	                                                                                           nnrt::core::functional::SortOrder::ASC);
	o3c::Tensor keys_gt_asc(
			std::vector<float>{
					1, 2, 3, 4, 5, // 0, 3, 1, 2, 4
					-2, -1, -1, 3, 4, // 7, 6, 9, 8, 5
					-4, -1, 0, 1, 2   // 12, 11, 10, 14, 13
			}, {3, 5}, o3c::Float32, device
	);
	o3c::Tensor values_gt_asc(std::vector<int>{
			0, 3, 1, 2, 4,
			7, 6, 9, 8, 5,
			12, 11, 10, 14, 13
	}, {3, 5}, o3c::Int32, device);

	REQUIRE(keys_asc.AllEqual(keys_gt_asc));
	REQUIRE(values_asc.AllEqual(values_gt_asc));

	o3c::Tensor values_asc_nnf, keys_asc_nnf;
	std::tie(values_asc_nnf, keys_asc_nnf) =
			nnrt::core::functional::SortTensorAlongLastDimensionByKey(value_tensor, key_tensor, true, nnrt::core::functional::SortOrder::ASC);

	o3c::Tensor keys_gt_asc_nnf(
			std::vector<float>{
					1, 2, 3, 4, 5,
					3, 4, -2, -1, -1, // 8, 5, 7, 6, 9,
					0, 1, 2, -4, -1 //  10, 14, 13, 12, 11
			}, {3, 5}, o3c::Float32, device
	);
	o3c::Tensor values_gt_asc_nnf(std::vector<int>{
			0, 3, 1, 2, 4,
			8, 5, 7, 6, 9,
			10, 14, 13, 12, 11
	}, {3, 5}, o3c::Int32, device);

	REQUIRE(keys_asc_nnf.AllEqual(keys_gt_asc_nnf));
	REQUIRE(values_asc_nnf.AllEqual(values_gt_asc_nnf));

	o3c::Tensor values_desc, keys_desc;
	std::tie(values_desc, keys_desc) = nnrt::core::functional::SortTensorAlongLastDimensionByKey(value_tensor, key_tensor, true,
	                                                                                             nnrt::core::functional::SortOrder::DESC);
	o3c::Tensor keys_gt_desc(
			std::vector<float>{
					5, 4, 3, 2, 1,
					4, 3, -1, -1, -2,
					2, 1, 0, -1, -4
			}, {3, 5}, o3c::Float32, device
	);
	o3c::Tensor values_gt_desc(std::vector<int>{
			4, 2, 1, 3, 0,
			5, 8, 6, 9, 7,
			13, 14, 10, 11, 12
	}, {3, 5}, o3c::Int32, device);

	REQUIRE(keys_desc.AllEqual(keys_gt_desc));
	REQUIRE(values_desc.AllEqual(values_gt_desc));

	o3c::Tensor values_desc_pf, keys_desc_pf;
	std::tie(values_desc_pf, keys_desc_pf) = nnrt::core::functional::SortTensorAlongLastDimensionByKey(value_tensor, key_tensor, false,
	                                                                                                   nnrt::core::functional::SortOrder::DESC);
	o3c::Tensor keys_gt_desc_pf(
			std::vector<float>{
					5, 4, 3, 2, 1,
					-1, -1, -2, 4, 3,
					-1, -4, 2, 1, 0
			}, {3, 5}, o3c::Float32, device
	);
	o3c::Tensor values_gt_desc_pf(std::vector<int>{
			4, 2, 1, 3, 0,
			6, 9, 7, 5, 8,
			11, 12, 13, 14, 10
	}, {3, 5}, o3c::Int32, device);

	REQUIRE(keys_desc_pf.AllEqual(keys_gt_desc_pf));
	REQUIRE(values_desc_pf.AllEqual(values_gt_desc_pf));
}


TEST_CASE("Test Sorting Along Last Dimension By Key - CPU") {
	auto device = o3c::Device("CPU:0");
	TestSortingAlongLastDimensionByKey(device);
}

TEST_CASE("Test Sorting Along Last Dimension By Key - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestSortingAlongLastDimensionByKey(device);
}