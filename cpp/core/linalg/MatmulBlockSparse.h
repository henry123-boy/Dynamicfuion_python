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
#pragma once
// third-party includes
#include <open3d/core/Tensor.h>
// local includes
#include "core/linalg/MatrixPreprocessingOperation.h"


namespace nnrt::core::linalg {


/**
 * \brief Product of all square blocks listed in (dense) array A by blocks in rows of block-sparse matrix B, with one A block per B row.
 * \details Equivalent to mathematical matrix multiplication DB, where D is a block-diagonal matrix consisting of blocks in A.
 * \param blocks_a dense array A - has to consist of square block matrices of the same size as ones in B
 * \param blocks_b list of blocks in the block-sparse matrix B, blocks must be the same size as ones in A
 * \param blocks_b_coordinates coordinates of blocks in B (in blocks, not scalar block coefficients)
 * \return tuple containing (1) list of resulting product blocks and (2) their coordinates
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates
);

/**
 * \brief Product of all square blocks listed in (dense) array A by blocks in rows of block-sparse matrix B, with one A block per B row.
 * \details In the non-padded part (read on), equivalent to mathematical matrix multiplication DB, where D is a block-diagonal matrix consisting of
 * blocks in A. If there are fewer blocks in A than rows in B, pads lower rows of output with zero blocks such that the output has blocks in every
 * position where B has blocks (i.e. blocks_b_coordinates can be also used to access output blocks).
 * \param blocks_a dense array A - has to consist of square block matrices of the same size as ones in B
 * \param blocks_b list of blocks in the block-sparse matrix B, blocks must be the same size as ones in A
 * \param blocks_b_coordinates coordinates of blocks in B (in blocks, not scalar block coefficients)
 * \return list of resulting product blocks,
 */
open3d::core::Tensor
MatmulBlockSparseRowWisePadded(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates
);

open3d::core::Tensor BlockSparseAndVectorProduct(
		const open3d::core::Tensor& blocks_a,
		int m,
		const open3d::core::Tensor& blocks_a_coordinates,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& vector_b
);


/**
 * \brief Compute product of block-sparse matrix a and block-sparse matrix b.
 * \details Preprocessing operations may be specified to transpose the argument matrices before product. Inner dimensions of argument matrices, after
 * any possible preprocessing, must match.
 * \param blocks_a blocks of matrix A
 * \param blocks_a_breadboard a rectangular matrix specifying the index of a block in A at each position (or -1 for no block). Must have type Int16.
 * \param matrix_a_preprocessing preprocessing op for matrix A.
 * \param blocks_b blocks of matrix B. Individual blocks must have same size as in A, but their count (leading dimension) may differ.
 * \param blocks_b_breadboard a rectangular matrix specifying the index of a block in B at each position (or -1 for no block). Must have type Int16.
 * \param matrix_b_preprocessing preprocessing op for matrix B.
 * \return
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparse(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_a_breadboard,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_breadboard,
		MatrixPreprocessingOperation matrix_b_preprocessing
);

namespace internal {

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates,
		bool padded
);

template<open3d::core::Device::DeviceType TDeviceType>
void BlockSparseAndVectorProduct(
		open3d::core::Tensor& out_vector,
		int m,
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_a_coordinates,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& vector_b
);

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparse(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& a_block_breadboard,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& b_block_breadboard,
		MatrixPreprocessingOperation matrix_b_preprocessing
);

} // namespace internal

} // namespace nnrt::core::linalg