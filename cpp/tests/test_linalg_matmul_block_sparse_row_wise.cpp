//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/23.
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
#include "core/linalg/MatmulBlockSparseRowWise.h"

namespace o3c = open3d::core;


void TestMatmulBlockSparseRowWise(const o3c::Device& device) {
	o3c::Tensor a_blocks(std::vector<float>{
			27., 26., 25.,
			24., 23., 22.,
			21., 20., 19.,

			18., 17., 16.,
			15., 14., 13.,
			12., 11., 10.,

			 9.,  8.,  7.,
			 6.,  5.,  4.,
			 3.,  2.,  1.
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor b_blocks(std::vector<float>{
			  0.,   1.,   2.,
			  3.,   4.,   5.,
			  6.,   7.,   8.,

			  9.,  10.,  11.,
			 12.,  13.,  14.,
			 15.,  16.,  17.,
			 

			 27.,  28.,  29.,
			 30.,  31.,  32.,
			 33.,  34.,  35.,

			 36.,  37.,  38.,
			 39.,  40.,  41.,
			 42.,  43.,  44.,

			 45.,  46.,  47.,
			 48.,  49.,  50.,
			 51.,  52.,  53.,

			 54.,  55.,  56.,
			 57.,  58.,  59.,
			 60.,  61.,  62.,

			 63.,  64.,  65.,
			 66.,  67.,  68.,
			 69.,  70.,  71.,

			 72.,  73.,  74.,
			 75.,  76.,  77.,
			 78.,  79.,  80.,

			 81.,  82.,  83.,
			 84.,  85.,  86.,
			 87.,  88.,  89.,

			 90.,  91.,  92.,
			 93.,  94.,  95.,
			 96.,  97.,  98.,

			 99., 100., 101.,
			102., 103., 104.,
			105., 106., 107.
	}, {11, 3, 3}, o3c::Float32, device);

	o3c::Tensor b_block_coordinates(std::vector<int>{
			0, 0,
			1, 0,
			3, 0,
			0, 1,
			1, 1,
			2, 1,
			3, 1,
			0, 2,
			1, 2,
			2, 2,
			3, 2
	}, {11, 2}, o3c::Int32, device);

	o3c::Tensor c_blocks_gt(std::vector<float>{
			 228.,  306.,  384.,
			 201.,  270.,  339.,
			 174.,  234.,  294.,

			 606.,  657.,  708.,
			 498.,  540.,  582.,
			 390.,  423.,  456.,

			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.,

			3036., 3114., 3192.,
			2685., 2754., 2823.,
			2334., 2394., 2454.,

			2442., 2493., 2544.,
			2010., 2052., 2094.,
			1578., 1611., 1644.,

			1362., 1386., 1410.,
			 849.,  864.,  879.,
			 336.,  342.,  348.,

			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.,

			5844., 5922., 6000.,
			5169., 5238., 5307.,
			4494., 4554., 4614.,

			4278., 4329., 4380.,
			3522., 3564., 3606.,
			2766., 2799., 2832.,

			2226., 2250., 2274.,
			1389., 1404., 1419.,
			 552.,  558.,  564.,

			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.,
			   // 0.,    0.,    0.
	}, {8, 3, 3}, o3c::Float32, device);

	o3c::Tensor c_block_coordinates_gt(std::vector<int>{
			0, 0,
			1, 0,
			// 3, 0,
			0, 1,
			1, 1,
			2, 1,
			// 3, 1,
			0, 2,
			1, 2,
			2, 2,
			// 3, 2
	}, {8, 2}, o3c::Int32, device);

	o3c::Tensor c_blocks_padded_gt(std::vector<float>{
			228.,  306.,  384.,
			201.,  270.,  339.,
			174.,  234.,  294.,

			606.,  657.,  708.,
			498.,  540.,  582.,
			390.,  423.,  456.,

			0.,    0.,    0.,
			0.,    0.,    0.,
			0.,    0.,    0.,

			3036., 3114., 3192.,
			2685., 2754., 2823.,
			2334., 2394., 2454.,

			2442., 2493., 2544.,
			2010., 2052., 2094.,
			1578., 1611., 1644.,

			1362., 1386., 1410.,
			849.,  864.,  879.,
			336.,  342.,  348.,

			0.,    0.,    0.,
			0.,    0.,    0.,
			0.,    0.,    0.,

			5844., 5922., 6000.,
			5169., 5238., 5307.,
			4494., 4554., 4614.,

			4278., 4329., 4380.,
			3522., 3564., 3606.,
			2766., 2799., 2832.,

			2226., 2250., 2274.,
			1389., 1404., 1419.,
			552.,  558.,  564.,

			0.,    0.,    0.,
			0.,    0.,    0.,
			0.,    0.,    0.
	}, {11, 3, 3}, o3c::Float32, device);


	o3c::Tensor c_blocks, c_block_coordinates;
	std::tie(c_blocks, c_block_coordinates) = nnrt::core::linalg::MatmulBlockSparseRowWise(a_blocks, b_blocks, b_block_coordinates);
	REQUIRE(c_blocks.AllClose(c_blocks_gt));
	REQUIRE(c_block_coordinates.AllClose(c_block_coordinates_gt));

	o3c::Tensor c_blocks_padded = nnrt::core::linalg::MatmulBlockSparseRowWisePadded(a_blocks, b_blocks, b_block_coordinates);
	REQUIRE(c_blocks_padded.AllClose(c_blocks_padded_gt));
}

TEST_CASE("Test Matmul Block Sparse Row-Wise - CPU") {
	auto device = o3c::Device("CPU:0");
	TestMatmulBlockSparseRowWise(device);
}

TEST_CASE("Test Matmul Block Sparse Row-Wise - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestMatmulBlockSparseRowWise(device);
}
