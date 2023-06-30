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
// 3rd-party
#include <open3d/core/Tensor.h>

namespace nnrt::core::linalg {


/**
 * \brief a symmetric matrix approximating a block-sparse arrow head structure, where diagonal consists of (non-zero) square blocks, and the
 * "arrowhead" consists of sparse blocks above and to the left of the bottom-right portion of the diagonal (may be multiple blocks wide above the diagonal)
 *
 * Conceptual sparsity diagram (unfilled spaces are all-zero blocks, D,A,B, and C are dense or partially-dense blocks):
 *
 * |D        B|
 * | D      B |
 * |  D       |
 * |   D    B |
 * |    D    B|
 * |     D  B |
 * |      D   |
 * |       D B|
 * | A A A  CC|
 * |A   A  ACC|
 *
 * D -- stem diagonal blocks
 * B -- upper right wing blocks
 * C -- arrow point, or "corner" blocks. These can be diagonal or non-diagonal.
 * A -- B^T for corresponding positions mirrored over the diagonal, or lower left wing blocks
 */
struct BlockSparseArrowheadMatrix {
	open3d::core::Tensor upper_right_wing_blocks;
	open3d::core::Tensor upper_right_wing_block_coordinates;
	open3d::core::Tensor upper_right_arrow_wing_breadboard;

	//TODO: not sure if these are needed at all
	open3d::core::Tensor upper_row_block_lists;
	open3d::core::Tensor upper_row_block_counts;
	open3d::core::Tensor upper_column_block_lists;
	open3d::core::Tensor upper_column_block_counts;

	open3d::core::Tensor stem_diagonal_blocks;
	open3d::core::Tensor corner_dense_matrix;
	// (first) index of the block along the diagonal at which the "arrowhead" is attached to the "stem" above and below
	int arrow_base_block_index;
};

} // nnrt::core::linalg
