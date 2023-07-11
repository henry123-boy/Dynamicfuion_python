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
#include "core/linalg/MatmulBlockSparse.h"

namespace o3c = open3d::core;


void TestMatmulBlockSparseRowWise_Small(const o3c::Device& device) {
	o3c::Tensor a_blocks(std::vector<float>{
			27., 26., 25.,
			24., 23., 22.,
			21., 20., 19.,

			18., 17., 16.,
			15., 14., 13.,
			12., 11., 10.,

			9., 8., 7.,
			6., 5., 4.,
			3., 2., 1.
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor b_blocks(std::vector<float>{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,

			9., 10., 11.,
			12., 13., 14.,
			15., 16., 17.,


			27., 28., 29.,
			30., 31., 32.,
			33., 34., 35.,

			36., 37., 38.,
			39., 40., 41.,
			42., 43., 44.,

			45., 46., 47.,
			48., 49., 50.,
			51., 52., 53.,

			54., 55., 56.,
			57., 58., 59.,
			60., 61., 62.,

			63., 64., 65.,
			66., 67., 68.,
			69., 70., 71.,

			72., 73., 74.,
			75., 76., 77.,
			78., 79., 80.,

			81., 82., 83.,
			84., 85., 86.,
			87., 88., 89.,

			90., 91., 92.,
			93., 94., 95.,
			96., 97., 98.,

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
			228., 306., 384.,
			201., 270., 339.,
			174., 234., 294.,

			606., 657., 708.,
			498., 540., 582.,
			390., 423., 456.,

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
			849., 864., 879.,
			336., 342., 348.,

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
			552., 558., 564.,

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
			228., 306., 384.,
			201., 270., 339.,
			174., 234., 294.,

			606., 657., 708.,
			498., 540., 582.,
			390., 423., 456.,

			0., 0., 0.,
			0., 0., 0.,
			0., 0., 0.,

			3036., 3114., 3192.,
			2685., 2754., 2823.,
			2334., 2394., 2454.,

			2442., 2493., 2544.,
			2010., 2052., 2094.,
			1578., 1611., 1644.,

			1362., 1386., 1410.,
			849., 864., 879.,
			336., 342., 348.,

			0., 0., 0.,
			0., 0., 0.,
			0., 0., 0.,

			5844., 5922., 6000.,
			5169., 5238., 5307.,
			4494., 4554., 4614.,

			4278., 4329., 4380.,
			3522., 3564., 3606.,
			2766., 2799., 2832.,

			2226., 2250., 2274.,
			1389., 1404., 1419.,
			552., 558., 564.,

			0., 0., 0.,
			0., 0., 0.,
			0., 0., 0.
	}, {11, 3, 3}, o3c::Float32, device);


	o3c::Tensor c_blocks, c_block_coordinates;
	std::tie(c_blocks, c_block_coordinates) = nnrt::core::linalg::MatmulBlockSparseRowWise(a_blocks, b_blocks, b_block_coordinates);
	REQUIRE(c_blocks.AllClose(c_blocks_gt));
	REQUIRE(c_block_coordinates.AllClose(c_block_coordinates_gt));

	o3c::Tensor c_blocks_padded = nnrt::core::linalg::MatmulBlockSparseRowWisePadded(a_blocks, b_blocks, b_block_coordinates);
	REQUIRE(c_blocks_padded.AllClose(c_blocks_padded_gt));
}

TEST_CASE("Test Matmul Block Sparse Row-Wise (Small) - CPU") {
	auto device = o3c::Device("CPU:0");
	TestMatmulBlockSparseRowWise_Small(device);
}

TEST_CASE("Test Matmul Block Sparse Row-Wise (Small) - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestMatmulBlockSparseRowWise_Small(device);
}


void TestMatmulBlockSparse_Small(const o3c::Device& device) {
	o3c::Tensor a_blocks(std::vector<float>{
			3., 4.,
			5., 6.,

			1., 2.,
			3., 4.,

			5., 6.,
			7., 8.
	}, {3, 2, 2}, o3c::Float32, device);

	// o3c::Tensor a_coordinates(std::vector<int32_t>{
	// 	0, 0,
	// 	0, 1,
	// 	1, 2
	// }, {3, 2}, o3c::Int32, device);

	// A =
	// [[3., 4., 1., 2., 0., 0.],
	//  [5., 6., 3., 4., 0., 0.],
	//  [0., 0., 0., 0., 5., 6.],
	//  [0., 0., 0., 0., 7., 8.]]

	o3c::Tensor a_breadboard(std::vector<int16_t>{
			0, 1, -1,
			-1, -1, 2
	}, {2, 3}, o3c::Int16, device);

	o3c::Tensor b_blocks(std::vector<float>{
			1., 2.,
			3., 4.,

			5., 6.,
			7., 8.,

			9., 10.,
			11., 0.
	}, {3, 2, 2}, o3c::Float32, device);

	// o3c::Tensor b_coordinates(std::vector<int32_t>{
	// 		0, 0,
	// 		1, 1,
	// 		2, 0
	// }, {3, 2}, o3c::Int32, device);

	// B =
	//  [[ 1.,  2.,  0.,  0.],
	//   [ 3.,  4.,  0.,  0.],
	//   [ 0.,  0.,  5.,  6.],
	//   [ 0.,  0.,  7.,  8.],
	//   [ 9., 10.,  0.,  0.],
	//   [11.,  0.,  0.,  0.]]


	o3c::Tensor b_breadboard(std::vector<int16_t>{
			0, -1,
			-1, 1,
			2, -1
	}, {3, 2}, o3c::Int16, device);

	o3c::Tensor c_blocks, c_block_coordinates;
	std::tie(c_blocks, c_block_coordinates) =
			nnrt::core::linalg::MatmulBlockSparse(a_blocks, a_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::NONE,
			                                      b_blocks, b_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::NONE);
	o3c::Tensor c_blocks_gt(std::vector<float>{
			15., 22.,
			23., 34.,

			19., 22.,
			43., 50.,

			111., 50.,
			151., 70.
	}, {3, 2, 2}, o3c::Float32, device);

	o3c::Tensor c_block_coordinates_gt(std::vector<int32_t>{
			0, 0,
			0, 1,
			1, 0
	}, {3, 2}, o3c::Int32, device);

	REQUIRE(c_blocks.AllClose(c_blocks_gt));
	REQUIRE(c_block_coordinates.AllClose(c_block_coordinates_gt));

	o3c::Tensor d_blocks, d_block_coordinates;
	std::tie(d_blocks, d_block_coordinates) =
			nnrt::core::linalg::MatmulBlockSparse(b_blocks, b_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::TRANSPOSE,
			                                      a_blocks, a_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::TRANSPOSE);

	o3c::Tensor d_blocks_gt(std::vector<float>{
			15., 23.,
			22., 34.,

			111., 151.,
			50., 70.,

			19., 43.,
			22., 50.
	}, {3, 2, 2}, o3c::Float32, device);

	o3c::Tensor d_block_coordinates_gt(std::vector<int32_t>{
			0, 0,
			0, 1,
			1, 0
	}, {3, 2}, o3c::Int32, device);

	REQUIRE(d_blocks.AllClose(d_blocks_gt));
	REQUIRE(d_block_coordinates.AllClose(d_block_coordinates_gt));

	o3c::Tensor e_blocks, e_block_coordinates;
	std::tie(e_blocks, e_block_coordinates) =
			nnrt::core::linalg::MatmulBlockSparse(b_blocks, b_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::TRANSPOSE,
			                                      b_blocks, b_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::NONE);

	o3c::Tensor e_blocks_gt(std::vector<float>{
			212., 104.,
			104., 120.,

			74., 86.,
			86., 100.
	}, {2, 2, 2}, o3c::Float32, device);

	o3c::Tensor e_block_coordinates_gt(std::vector<int32_t>{
			0, 0,
			1, 1
	}, {2, 2}, o3c::Int32, device);

	REQUIRE(e_blocks.AllClose(e_blocks_gt));
	REQUIRE(e_block_coordinates.AllClose(e_block_coordinates_gt));

	o3c::Tensor f_blocks, f_block_coordinates;
	std::tie(f_blocks, f_block_coordinates) =
			nnrt::core::linalg::MatmulBlockSparse(a_blocks, a_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::NONE,
			                                      a_blocks, a_breadboard, nnrt::core::linalg::MatrixPreprocessingOperation::TRANSPOSE);

	o3c::Tensor f_blocks_gt(std::vector<float>{
			30., 50.,
			50., 86.,

			61., 83.,
			83., 113.
	}, {2, 2, 2}, o3c::Float32, device);

	o3c::Tensor f_block_coordinates_gt(std::vector<int32_t>{
			0, 0,
			1, 1
	}, {2, 2}, o3c::Int32, device);

	REQUIRE(f_blocks.AllClose(f_blocks_gt));
	REQUIRE(f_block_coordinates.AllClose(f_block_coordinates_gt));


}

TEST_CASE("Test Matmul Block Sparse (Small) - CPU") {
	auto device = o3c::Device("CPU:0");
	TestMatmulBlockSparse_Small(device);
}

TEST_CASE("Test Matmul Block Sparse (Small) - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestMatmulBlockSparse_Small(device);
}

void TestBlockSparseAndVectorProduct_Small(const o3c::Device& device) {
	int m = 4;
	o3c::Tensor a_blocks(std::vector<float>{
			3., 4.,
			5., 6.,

			1., 2.,
			3., 4.,

			5., 6.,
			7., 8.
	}, {3, 2, 2}, o3c::Float32, device);

	o3c::Tensor a_coordinates(std::vector<int32_t>{
			0, 0,
			0, 1,
			1, 2
	}, {3, 2}, o3c::Int32, device);

	// A =
	// [[3., 4., 1., 2., 0., 0.],
	//  [5., 6., 3., 4., 0., 0.],
	//  [0., 0., 0., 0., 5., 6.],
	//  [0., 0., 0., 0., 7., 8.]]

	o3c::Tensor vector_b(std::vector<float>{
			-2.,
			-1.,
			0.,
			1.,
			2.,
			3.,
	}, {6}, o3c::Float32, device);

	o3c::Tensor vector_c_gt(std::vector<float>{
			-8.,
			-12.,
			28.,
			38.,
	}, {4}, o3c::Float32, device);

	o3c::Tensor vector_c =
			nnrt::core::linalg::BlockSparseAndVectorProduct(
					a_blocks, m, a_coordinates, std::tuple<int32_t, int32_t>(), nnrt::core::linalg::MatrixPreprocessingOperation::NONE, vector_b
			);


	REQUIRE(vector_c.AllClose(vector_c_gt));

	o3c::Tensor b_blocks(std::vector<float>{
			1., 2.,
			3., 4.,

			5., 6.,
			7., 8.,

			9., 10.,
			11., 0.
	}, {3, 2, 2}, o3c::Float32, device);

	o3c::Tensor b_coordinates(std::vector<int32_t>{
			0, 0,
			1, 1,
			2, 0
	}, {3, 2}, o3c::Int32, device);

	// B =
	//  [[ 1.,  2.,  0.,  0.],
	//   [ 3.,  4.,  0.,  0.],
	//   [ 0.,  0.,  5.,  6.],
	//   [ 0.,  0.,  7.,  8.],
	//   [ 9., 10.,  0.,  0.],
	//   [11.,  0.,  0.,  0.]]

	o3c::Tensor vector_d_gt(std::vector<float>{
			46.,
			12.,
			7.,
			8.,
	}, {4}, o3c::Float32, device);

	o3c::Tensor vector_d =
			nnrt::core::linalg::BlockSparseAndVectorProduct(
					b_blocks, m, b_coordinates, std::tuple<int32_t, int32_t>(), nnrt::core::linalg::MatrixPreprocessingOperation::TRANSPOSE, vector_b
			);

	REQUIRE(vector_d.AllClose(vector_d_gt));

}

TEST_CASE("Test Block-Sparse & Vector Product (Small) - CPU") {
	auto device = o3c::Device("CPU:0");
	TestBlockSparseAndVectorProduct_Small(device);
}

TEST_CASE("Test  Block-Sparse & Vector Product (Small) - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestBlockSparseAndVectorProduct_Small(device);
}


void TestDiagonalBlockSparseAndVectorProduct_Small(const o3c::Device& device) {
	o3c::Tensor d_blocks(std::vector<float>{
			2., 2.,
			2., 2.,

			1., 1.,
			1., 1.,

			3., 3.,
			3., 3.
	}, {3, 2, 2}, o3c::Float32, device);

	o3c::Tensor vector_b(std::vector<float>{
			-2.,
			-1.,
			0.,
			1.,
			2.,
			3.,
	}, {6}, o3c::Float32, device);

	o3c::Tensor vector_c_gt(std::vector<float>{
			-6.,
			-6.,
			1.,
			1.,
			15.,
			15.,
	}, {6}, o3c::Float32, device);

	o3c::Tensor vector_c = nnrt::core::linalg::DiagonalBlockSparseAndVectorProduct(d_blocks, vector_b);
	REQUIRE(vector_c.AllClose(vector_c_gt));
}

TEST_CASE("Test Diagonal Block-Sparse & Vector Product (Small) - CPU") {
	auto device = o3c::Device("CPU:0");
	TestDiagonalBlockSparseAndVectorProduct_Small(device);
}

TEST_CASE("Test  Diagonal Block-Sparse & Vector Product (Small) - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestDiagonalBlockSparseAndVectorProduct_Small(device);
}