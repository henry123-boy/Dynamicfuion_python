//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/23.
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
#include "SolveBlockSparseArrowheadCholesky.h"
#include "core/linalg/FactorizeBlocksCholesky.h"
#include "core/linalg/InvertBlocks.h"
#include "core/linalg/Matmul3D.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg {

void SolveBlockSparseArrowheadCholesky(
		open3d::core::Tensor& X,
		const nnrt::core::linalg::BlockSparseArrowheadMatrix& A,
		const open3d::core::Tensor& B
) {
	o3c::Tensor L_diagonal_upper_left;
	FactorizeBlocksCholesky(L_diagonal_upper_left, A.diagonal_blocks.Slice(0, 0, A.arrow_base_block_index), UpLoTriangular::LOWER);
	o3c::Tensor L_inv_diagonal_upper_left = InvertTriangularBlocks(L_diagonal_upper_left, UpLoTriangular::LOWER);

	o3c::Tensor U_diagonal_upper_left = L_diagonal_upper_left.Transpose(1, 2);
	o3c::Tensor U_upper_right;



}

} // namespace nnrt::core::linalg

