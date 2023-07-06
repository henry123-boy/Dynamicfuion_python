//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/30/23.
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

// local includes
#include "SchurComplement.h"
#include "InvertBlocks.h"
#include "MatmulBlockSparse.h"
#include "DiagonalBlocks.h"
#include "SparseBlocks.h"


namespace o3c = open3d::core;
namespace nnrt::core::linalg {


open3d::core::Tensor ComputeSchurComplementOfArrowStem_Dense(const BlockSparseArrowheadMatrix& a) {
	// compute D^(-1)
	o3c::Tensor inverted_stem = InvertSymmetricPositiveDefiniteBlocks(a.StemDiagonalBlocks());

	// compute D^(-1)B
	o3c::Tensor inverted_stem_and_upper_wing_product_blocks =
			core::linalg::MatmulBlockSparseRowWisePadded(inverted_stem, a.upper_wing_blocks, a.upper_wing_block_coordinates);

	return ComputeSchurComplementOfArrowStem_Dense(a, inverted_stem_and_upper_wing_product_blocks);
}

open3d::core::Tensor
linalg::ComputeSchurComplementOfArrowStem_Dense(const BlockSparseArrowheadMatrix& a, const open3d::core::Tensor& inverted_stem_and_upper_wing_product_blocks) {
	// Formula for Schur complement of arrow stem (D) for matrix A:
	// A/D = C - B^(T)D^(-1)B
	o3c::Device device = a.GetDevice();
	int64_t corner_size_blocks = a.diagonal_block_count - a.arrow_base_block_index;
	int64_t block_size = a.GetBlockSize();

	o3c::Tensor corner_dense_matrix =
			o3c::Tensor::Zeros({corner_size_blocks * block_size, corner_size_blocks * block_size}, o3c::Float32, device);

	// compute dense C:
	core::linalg::FillInDiagonalBlocks(corner_dense_matrix, a.CornerDiagonalBlocks());
	core::linalg::FillInSparseBlocks(corner_dense_matrix, a.corner_upper_blocks, a.corner_upper_block_coordinates,
	                                 std::make_tuple(static_cast<int64_t>(-a.arrow_base_block_index),
	                                                 static_cast<int64_t>(-a.arrow_base_block_index)),
	                                 false);
	//TODO: not sure if the lower-diagonal blocks are used at all during dense cholesky factorization, so try without filling
	// them, i.e. without below line.
	core::linalg::FillInSparseBlocks(corner_dense_matrix, a.corner_upper_blocks, a.corner_upper_block_coordinates,
	                                 std::make_tuple(static_cast<int64_t>(-a.arrow_base_block_index),
	                                                 static_cast<int64_t>(-a.arrow_base_block_index)),
	                                 true);


	// compute sparse, then dense B^(T)D^(-1)B
	o3c::Tensor rhs_operand_blocks, rhs_operand_block_coordinates;
	std::tie(rhs_operand_blocks, rhs_operand_block_coordinates) =
			core::linalg::MatmulBlockSparse(a.upper_wing_blocks, a.upper_wing_breadboard, MatrixPreprocessingOperation::TRANSPOSE,
			                                inverted_stem_and_upper_wing_product_blocks, a.upper_wing_breadboard, MatrixPreprocessingOperation::NONE);

	// compute C - B^(T)D^(-1)B
	core::linalg::SubtractSparseBlocks(corner_dense_matrix, rhs_operand_blocks, rhs_operand_block_coordinates, std::make_tuple(0LL, 0LL), false);

	return corner_dense_matrix;
}

} // namespace nnrt::core::linalg