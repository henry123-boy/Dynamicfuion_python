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
#include "core/linalg/SolveBlockDiagonalCholesky.h"
#include "core/linalg/FactorizeBlocksCholesky.h"
#include "core/linalg/InvertBlocks.h"
#include "core/TensorManipulationRoutines.h"

namespace o3c = open3d::core;

void TestCholeskyBlockDiagonalSolver(const o3c::Device& device) {
	o3c::Tensor A_blocks(std::vector<float>{
			7.66466999, 7.42160096, 7.96971846, 5.41618416, 5.48901906, 6.29302529,
			7.42160096, 11.28136639, 10.02191478, 7.6142696, 6.11965727, 8.29031205,
			7.96971846, 10.02191478, 12.84076044, 7.99068493, 7.71414652, 8.53580411,
			5.41618416, 7.6142696, 7.99068493, 8.59449478, 5.25876695, 6.28306648,
			5.48901906, 6.11965727, 7.71414652, 5.25876695, 6.5656741, 5.7888223,
			6.29302529, 8.29031205, 8.53580411, 6.28306648, 5.7888223, 8.58118284,

			11.02955047, 7.99694855, 8.60120371, 7.94013951, 8.92082018, 6.28801604,
			7.99694855, 9.59915016, 8.38727439, 7.69725731, 8.57161899, 6.67614202,
			8.60120371, 8.38727439, 10.27710871, 7.29562432, 9.01899879, 6.63105238,
			7.94013951, 7.69725731, 7.29562432, 9.06157844, 8.29751497, 5.44649512,
			8.92082018, 8.57161899, 9.01899879, 8.29751497, 11.55352825, 6.84668649,
			6.28801604, 6.67614202, 6.63105238, 5.44649512, 6.84668649, 6.56172847,

			9.76694034, 7.01864966, 6.48232141, 7.00072462, 7.35419513, 5.94435122,
			7.01864966, 9.59364701, 6.5592933, 7.44435127, 7.66117909, 6.28624863,
			6.48232141, 6.5592933, 10.01902388, 6.28058687, 7.16248224, 6.99130877,
			7.00072462, 7.44435127, 6.28058687, 8.7593816, 8.17335754, 6.52806757,
			7.35419513, 7.66117909, 7.16248224, 8.17335754, 11.15212005, 6.81745422,
			5.94435122, 6.28624863, 6.99130877, 6.52806757, 6.81745422, 9.11671782
	}, {3, 6, 6}, o3c::Float32, device);

	o3c::Tensor B(std::vector<float>{
			0.45293155, 0.52475495,
			0.21761689, 0.54381511,
			0.17009712, 0.00117492,
			0.72334337, 0.83178217,
			0.82902052, 0.62671342,
			0.59249606, 0.7532337,

			0.41178043, 0.74776442,
			0.32215233, 0.54746278,
			0.37314376, 0.05380477,
			0.06205624, 0.37561555,
			0.77131601, 0.40215776,
			0.02648144, 0.15254462,

			0.80500491, 0.25057921,
			0.21371886, 0.43475709,
			0.39716974, 0.15878393,
			0.1928535, 0.65798109,
			0.46133341, 0.05955819,
			0.95671584, 0.59674103
	}, {18, 2}, o3c::Float32, device);

	o3c::Tensor expected_X(std::vector<float>{
			0.03432138, 0.05864467,
			-0.09385886, 0.00502414,
			-0.26434026, -0.35847137,
			0.13916333, 0.15648315,
			0.30488228, 0.21968781,
			0.0899299, 0.13371622,
			0.05101893, 0.18088418,
			0.10856187, 0.21007842,
			-0.02048628, -0.2150583,
			-0.19654963, -0.09012525,
			0.22811848, 0.03767047,
			-0.20948814, -0.1110012,
			0.15458118, -0.03555937,
			-0.06289044, 0.00555059,
			-0.078527, -0.04738612,
			-0.18840044, 0.211494,
			0.04203119, -0.14935129,
			0.21120842, 0.08139612
	}, {18, 2}, o3c::Float32, device);

	o3c::Tensor X;
	nnrt::core::linalg::SolveBlockDiagonalCholesky(X, A_blocks, B);

	REQUIRE(expected_X.AllClose(X));
}

TEST_CASE("Test Solve Cholesky Block Diagonal - CPU") {
	auto device = o3c::Device("CPU:0");
	TestCholeskyBlockDiagonalSolver(device);
}

TEST_CASE("Test Solve Cholesky Block Diagonal - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestCholeskyBlockDiagonalSolver(device);
}

void TestBlockCholeskyFactorization(const o3c::Device& device) {
	o3c::Tensor blocks(std::vector<float>{
			7.66466999, 7.42160096, 7.96971846, 5.41618416, 5.48901906, 6.29302529,
			7.42160096, 11.28136639, 10.02191478, 7.6142696, 6.11965727, 8.29031205,
			7.96971846, 10.02191478, 12.84076044, 7.99068493, 7.71414652, 8.53580411,
			5.41618416, 7.6142696, 7.99068493, 8.59449478, 5.25876695, 6.28306648,
			5.48901906, 6.11965727, 7.71414652, 5.25876695, 6.5656741, 5.7888223,
			6.29302529, 8.29031205, 8.53580411, 6.28306648, 5.7888223, 8.58118284,

			11.02955047, 7.99694855, 8.60120371, 7.94013951, 8.92082018, 6.28801604,
			7.99694855, 9.59915016, 8.38727439, 7.69725731, 8.57161899, 6.67614202,
			8.60120371, 8.38727439, 10.27710871, 7.29562432, 9.01899879, 6.63105238,
			7.94013951, 7.69725731, 7.29562432, 9.06157844, 8.29751497, 5.44649512,
			8.92082018, 8.57161899, 9.01899879, 8.29751497, 11.55352825, 6.84668649,
			6.28801604, 6.67614202, 6.63105238, 5.44649512, 6.84668649, 6.56172847,

			9.76694034, 7.01864966, 6.48232141, 7.00072462, 7.35419513, 5.94435122,
			7.01864966, 9.59364701, 6.5592933, 7.44435127, 7.66117909, 6.28624863,
			6.48232141, 6.5592933, 10.01902388, 6.28058687, 7.16248224, 6.99130877,
			7.00072462, 7.44435127, 6.28058687, 8.7593816, 8.17335754, 6.52806757,
			7.35419513, 7.66117909, 7.16248224, 8.17335754, 11.15212005, 6.81745422,
			5.94435122, 6.28624863, 6.99130877, 6.52806757, 6.81745422, 9.11671782
	}, {3, 6, 6}, o3c::Float32, device);
	o3c::Tensor factorized_blocks;
	nnrt::core::linalg::FactorizeBlocksCholesky_LowerTriangular(factorized_blocks, blocks);
	o3c::Tensor factorized_blocks_cpu = factorized_blocks.To(o3c::Device("CPU:0"));
	auto factorized_blocks_data = factorized_blocks_cpu.ToFlatVector<float>();
	for (int i_block = 0; i_block < 3; i_block++) {
		for (int i_row = 0; i_row < 6; i_row++) {
			for (int i_col = 0; i_col < 6; i_col++) {
				if (i_col > i_row) {
					factorized_blocks_data[i_block * 36 + i_row * 6 + i_col] = 0.0;
				}
			}
		}
	}
	o3c::Tensor factorized_blocks_ut_zeroed(factorized_blocks_data, {3, 6, 6}, o3c::Float32, o3c::Device("CPU:0"));

	o3c::Tensor factorized_blocks_gt(std::vector<float>{
			2.76851404, 0., 0., 0., 0., 0.,
			2.68071639, 2.02364177, 0., 0., 0., 0.,
			2.87869895, 1.13900561, 1.80458278, 0., 0., 0.,
			1.95635062, 1.171081, 0.56803857, 1.75302268, 0., 0.,
			1.98265892, 0.39765487, 0.86099527, 0.24256751, 1.29478047, 0.,
			2.27306967, 1.08559576, 0.41883431, 0.18648396, 0.34334677, 1.38120727,

			3.3210767, 0., 0., 0., 0., 0.,
			2.40793853, 1.94961078, 0., 0., 0., 0.,
			2.58988409, 1.1032934, 1.53373818, 0., 0., 0.,
			2.39083292, 0.99521331, 0.00367201, 1.5346118, 0., 0.,
			2.6861229, 1.07898468, 0.56842502, 0.5210026, 1.60608635, 0.,
			1.8933667, 1.08587386, 0.34517927, -0.1056762, 0.27898787, 1.26080075,

			3.12521045, 0., 0., 0., 0., 0.,
			2.24581665, 2.13306225, 0., 0., 0., 0.,
			2.07420317, 0.89121322, 2.21865817, 0., 0., 0.,
			2.24008102, 1.13149065, 0.28206431, 1.5432392, 0., 0.,
			2.35318397, 1.11406001, 0.58081754, 0.95750445, 1.76616867, 0.,
			1.9020643, 0.94444546, 0.99354588, 0.59512121, 0.08067067, 1.80529265
	}, {3, 6, 6}, o3c::Float32, o3c::Device("CPU:0"));

	REQUIRE(factorized_blocks_ut_zeroed.AllClose(factorized_blocks_gt, 1e-4));

}

TEST_CASE("Test Factorize Cholesky Blocks - CPU") {
	auto device = o3c::Device("CPU:0");
	TestBlockCholeskyFactorization(device);
}

TEST_CASE("Test Factorize Cholesky Blocks - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestBlockCholeskyFactorization(device);
}

void TestInvertTriangularBlocks_Lower(const o3c::Device& device) {
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

	o3c::Tensor inverted_blocks = nnrt::core::linalg::InvertTriangularBlocks(blocks, nnrt::core::linalg::UpLoTriangular::LOWER);

	o3c::Tensor inverted_blocks_gt(std::vector<float>{
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
	//__DEBUG
	auto inverted_blocks_cpu = inverted_blocks.To(o3c::Device("CPU:0"));
	auto inverted_blocks_gt_cpu = inverted_blocks_gt.To(o3c::Device("CPU:0"));

	REQUIRE(inverted_blocks.AllClose(inverted_blocks_gt, 1e-4));
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
	o3c::Tensor blocks(std::vector<float>{
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

	o3c::Tensor inverted_blocks = nnrt::core::linalg::InvertTriangularBlocks(blocks, nnrt::core::linalg::UpLoTriangular::UPPER);

	o3c::Tensor inverted_blocks_gt(std::vector<float>{
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
	//__DEBUG
	auto inverted_blocks_cpu = inverted_blocks.To(o3c::Device("CPU:0"));
	auto inverted_blocks_gt_cpu = inverted_blocks_gt.To(o3c::Device("CPU:0"));

	REQUIRE(inverted_blocks.AllClose(inverted_blocks_gt, 1e-4));

}

TEST_CASE("Test Invert Triangular Blocks Upper - CPU") {
	auto device = o3c::Device("CPU:0");
	TestInvertTriangularBlocks_Upper(device);
}

TEST_CASE("Test Invert Triangular Blocks Upper - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestInvertTriangularBlocks_Upper(device);
}

