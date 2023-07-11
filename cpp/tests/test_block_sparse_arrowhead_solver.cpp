//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/16/23.
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
// third-party includes
#include <open3d/core/Tensor.h>

// local includes
#include "test_main.hpp"
#include "test_utils/test_utils.hpp"

// code being tested
#include "core/linalg/SolveBlockSparseArrowheadCholesky.h"
#include "core/linalg/ZeroOutTriangularBlocks.h"
#include "core/linalg/DiagonalBlocks.h"
#include "core/linalg/SchurComplement.h"

namespace o3c = open3d::core;

nnrt::core::linalg::BlockSparseArrowheadMatrix LoadSparseArrowheadInputs(const o3c::Device& device) {
	nnrt::core::linalg::BlockSparseArrowheadMatrix matrix;
	matrix.arrow_base_block_index = 208;
	matrix.diagonal_block_count = 249;
	matrix.SetDiagonalBlocks(
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/diagonal_blocks.npy")
					.To(device).To(o3c::Float32)
	);
	matrix.upper_wing_blocks =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_wing_blocks.npy").To(device).To(o3c::Float32);
	matrix.upper_wing_block_coordinates =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_wing_block_coordinates.npy").To(device);
	matrix.upper_wing_breadboard = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_wing_breadboard.npy").To(device);
	matrix.corner_upper_blocks =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/corner_upper_blocks.npy").To(device).To(o3c::Float32);
	matrix.corner_upper_block_coordinates =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/corner_upper_block_coordinates.npy").To(device);

	return matrix;
}

void TestSchurComplementComputation(const o3c::Device& device) {
	auto matrix = LoadSparseArrowheadInputs(device);

	o3c::Tensor schur_complement_gt =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/stem_schur.npy").To(device).To(o3c::Float32);

	o3c::Tensor schur_complement = nnrt::core::linalg::ComputeSchurComplementOfArrowStem_Dense(matrix);

	REQUIRE(schur_complement.AllClose(schur_complement_gt, 0, 1e-6));
}

TEST_CASE("Test Block-Sparse Arrowhead Stem Schur CPU") {
	auto device = o3c::Device("CPU:0");
	TestSchurComplementComputation(device);
}

TEST_CASE("Test Block-Sparse Arrowhead Stem Schur CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestSchurComplementComputation(device);
}

void TestBlockSparseArrowheadSolver(const o3c::Device& device) {
	auto a = LoadSparseArrowheadInputs(device);

	o3c::Tensor b =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/b.npy").To(device).To(o3c::Float32);

	o3c::Tensor x_gt =
			o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/x.npy").To(device).To(o3c::Float32);

	o3c::Tensor x;
	nnrt::core::linalg::SolveBlockSparseArrowheadCholesky(x, a, b);

	REQUIRE(x.AllClose(x_gt, 0, 1e-6));
}

TEST_CASE("Test Block-Sparse Arrowhead Solver CPU") {
	auto device = o3c::Device("CPU:0");
	TestSchurComplementComputation(device);
}

TEST_CASE("Test Block-Sparse Arrowhead Solver CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestSchurComplementComputation(device);
}