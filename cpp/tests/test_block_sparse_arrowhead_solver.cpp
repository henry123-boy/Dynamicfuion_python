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

namespace o3c = open3d::core;

nnrt::core::linalg::BlockSparseArrowheadMatrix LoadSparseArrowheadInputs(const o3c::Device& device) {
	nnrt::core::linalg::BlockSparseArrowheadMatrix matrix;
	matrix.stem_diagonal_blocks = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/diagonal_blocks.npy").To(device)
	                                                                                                                              .To(o3c::Float32);
	matrix.wing_upper_blocks = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_blocks.npy").To(device).To(o3c::Float32);
	matrix.wing_upper_block_coordinates = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_block_coordinates.npy")
			.To(device);
	matrix.wing_upper_breadboard = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/breadboard.npy").To(device);
	matrix.upper_column_block_lists = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_column_block_lists.npy")
			.To(device);
	matrix.upper_column_block_counts = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_column_block_counts.npy")
			.To(device);
	matrix.upper_row_block_lists = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_row_block_lists.npy").To(device);
	matrix.upper_row_block_counts = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/upper_row_block_counts.npy")
			.To(device);
	matrix.arrow_base_block_index = 208;
	return matrix;
}

void TestCholeskyBlockSparseArrowheadFactorization(const o3c::Device& device) {
	auto matrix = LoadSparseArrowheadInputs(device);

	o3c::Tensor U_diag, U_upper, U_lower_right_dense;
	std::tie(U_diag, U_upper, U_lower_right_dense) = nnrt::core::linalg::FactorizeBlockSparseArrowheadCholesky_Upper(matrix);

	o3c::Tensor U_diag_gt = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/U_diag_upper_left.npy").To(device)
	                                                                                                                          .To(o3c::Float32);
	o3c::Tensor U_upper_gt = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/U_upper.npy").To(device).To(o3c::Float32);
	o3c::Tensor U_lower_right_dense_gt = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "/U_lower_right_dense.npy")
			.To(device).To(o3c::Float32);

	U_diag = U_diag.Contiguous();
	nnrt::core::linalg::ZeroOutTriangularBlocks(U_diag, nnrt::core::linalg::UpLoTriangular::LOWER);


	int block_size = static_cast<int32_t>(U_diag.GetShape(1));
	o3c::Tensor U_diag_corner = nnrt::core::linalg::GetDiagonalBlocks(U_lower_right_dense, block_size);
	nnrt::core::linalg::ZeroOutTriangularBlocks(U_diag_corner, nnrt::core::linalg::UpLoTriangular::LOWER);
	nnrt::core::linalg::FillInDiagonalBlocks(U_lower_right_dense, U_diag_corner);

	o3c::Tensor U_diag_corner_gt = nnrt::core::linalg::GetDiagonalBlocks(U_lower_right_dense_gt, block_size);


	REQUIRE(U_diag.AllClose(U_diag_gt));
	REQUIRE(U_upper.AllClose(U_upper_gt, 0, 1e-6));
	REQUIRE(U_diag_corner.AllClose(U_diag_corner_gt, 0, 1e-6));
	REQUIRE(U_lower_right_dense.AllClose(U_lower_right_dense_gt, 0, 1e-6));
}

TEST_CASE("Test Factorize Block-Sparse Arrowhead CPU") {
	auto device = o3c::Device("CPU:0");
	TestCholeskyBlockSparseArrowheadFactorization(device);
}

TEST_CASE("Test Factorize Block-Sparse Arrowhead CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestCholeskyBlockSparseArrowheadFactorization(device);
}
