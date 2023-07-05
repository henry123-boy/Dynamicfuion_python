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

namespace internal {

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates,
		bool padded
);

} // namespace internal

} // namespace nnrt::core::linalg