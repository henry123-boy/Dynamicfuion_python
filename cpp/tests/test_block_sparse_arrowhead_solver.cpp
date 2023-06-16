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

namespace o3c = open3d::core;

nnrt::core::linalg::BlockSparseArrowheadMatrix LoadSparseArrowheadInputs(){
	nnrt::core::linalg::BlockSparseArrowheadMatrix matrix;
	matrix.upper_block_breadboard = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "breadboard.npy");
	matrix.upper_column_block_lists = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "upper_column_block_lists.npy");
	matrix.upper_column_block_counts = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "upper_column_block_counts.npy");
	matrix.upper_row_block_lists = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "upper_row_block_lists.npy");
	matrix.upper_row_block_counts = o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "upper_row_block_counts.npy");

}

void TestCholeskyBlockArrowheadFactorization(const o3c::Device& device) {
	o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "U_diag_upper_left.npy");
	o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "U_upper_right.npy");
	o3c::Tensor::Load(test::generated_array_test_data_directory.ToString() + "U_lower_right_dense.npy");
}
