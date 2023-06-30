//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/24/23.
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
#include "core/linalg/TransposeBlocks.h"
#include "core/linalg/InvertBlocks.h"

namespace o3c = open3d::core;

void TestInvertTriangularBlocks_Lower(const o3c::Device& device) {
	o3c::Tensor blocks_lower(std::vector<float>{
			1, 0, 0,
			2, 3, 0,
			4, 5, 6,

			7, 0, 0,
			8, 9, 0,
			10, 11, 12,

			13, 0, 0,
			14, 15, 0,
			16, 17, 18
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor inverted_blocks_lower = nnrt::core::linalg::InvertTriangularBlocks(blocks_lower, nnrt::core::linalg::UpLoTriangular::LOWER);

	o3c::Tensor inverted_blocks_lower_gt(std::vector<float>{
			1., -0., 0.,
			-0.6666667, 0.33333334, -0.,
			-0.11111111, -0.2777778, 0.16666667,

			0.14285715, 0., -0.,
			-0.12698413, 0.11111111, 0.,
			-0.0026455, -0.10185185, 0.08333334,

			0.07692308, 0., 0.,
			-0.07179487, 0.06666667, 0.,
			-0.0005698, -0.06296296, 0.05555556
	}, {3, 3, 3}, o3c::Float32, device);

	REQUIRE(inverted_blocks_lower.AllClose(inverted_blocks_lower_gt, 1e-4));
}

TEST_CASE("Test Invert Triangular Blocks Lower - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInvertTriangularBlocks_Lower(device);
}

TEST_CASE("Test Invert Triangular Blocks Lower - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInvertTriangularBlocks_Lower(device);
}


void TestInvertTriangularBlocks_Upper(const o3c::Device& device) {
	o3c::Tensor blocks_upper(std::vector<float>{
			4, 5, 6,
			0, 2, 3,
			0, 0, 1,

			10, 11, 12,
			0, 8, 9,
			0, 0, 7,

			16, 17, 18,
			0, 14, 15,
			0, 0, 13
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor inverted_blocks_upper = nnrt::core::linalg::InvertTriangularBlocks(blocks_upper, nnrt::core::linalg::UpLoTriangular::UPPER);

	o3c::Tensor inverted_blocks_upper_gt(std::vector<float>{
			0.25, -0.625, 0.375,
			0., 0.5, -1.5,
			0., 0., 1.,

			0.1, -0.1375, 0.00535714,
			0., 0.125, -0.16071428,
			0., 0., 0.14285715,

			0.0625, -0.07589286, 0.00103022,
			0., 0.07142857, -0.08241758,
			0., 0., 0.07692308
	}, {3, 3, 3}, o3c::Float32, device);

	REQUIRE(inverted_blocks_upper.AllClose(inverted_blocks_upper_gt, 1e-4));

}

TEST_CASE("Test Invert Triangular Blocks Upper - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInvertTriangularBlocks_Upper(device);
}

TEST_CASE("Test Invert Triangular Blocks Upper - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInvertTriangularBlocks_Upper(device);
}

void TestTransposeBlocks(const o3c::Device& device) {
	o3c::Tensor blocks(std::vector<float>{
			1, 0, 0,
			2, 3, 0,
			4, 5, 6,

			7, 0, 0,
			8, 9, 0,
			10, 11, 12,

			13, 0, 0,
			14, 15, 0,
			16, 17, 18
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor blocks_transposed_gt(std::vector<float>{
			1, 2, 4,
			0, 3, 5,
			0, 0, 6,

			7, 8, 10,
			0, 9, 11,
			0, 0, 12,

			13, 14, 16,
			0, 15, 17,
			0, 0, 18
	}, {3, 3, 3}, o3c::Float32, device);

	nnrt::core::linalg::TransposeBlocksInPlace(blocks);

	REQUIRE(blocks.AllClose(blocks_transposed_gt, 1e-7));
}

TEST_CASE("Test Transpose Blocks - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestTransposeBlocks(device);
}


void TestInvertBlocks(const o3c::Device& device) {
	o3c::Tensor blocks(std::vector<float>{
			 16,  20,  24,
			 20,  29,  36,
			 24,  36,  46,

			100, 110, 120,
			110, 185, 204,
			120, 204, 274,

			256, 272, 288,
			272, 485, 516,
			288, 516, 718
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor inverted_blocks = nnrt::core::linalg::InvertSymmetricPositiveDefiniteBlocks(blocks);

	o3c::Tensor inverted_blocks_gt(std::vector<float>{
			 0.59375   , -0.875     ,  0.375     ,
			-0.875     ,  2.5       , -1.5       ,
			 0.375     , -1.5       ,  1.        ,

			 0.02893495, -0.01804847,  0.00076531,
			-0.01804847,  0.04145408, -0.02295918,
			 0.00076531, -0.02295918,  0.02040816,

			 0.00966704, -0.00550583,  0.00007925,
			-0.00550583,  0.0118947 , -0.00633981,
			 0.00007925, -0.00633981,  0.00591716
	}, {3, 3, 3}, o3c::Float32, device);

	REQUIRE(inverted_blocks.AllClose(inverted_blocks_gt, 1e-4));

}

TEST_CASE("Test Invert Blocks - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInvertBlocks(device);
}

TEST_CASE("Test Invert Blocks - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInvertBlocks(device);
}