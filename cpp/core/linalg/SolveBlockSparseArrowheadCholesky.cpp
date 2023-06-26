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
#include "core/linalg/MatmulBlockSparseRowWise.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg {

void SolveBlockSparseArrowheadCholesky(
		open3d::core::Tensor& X,
		const nnrt::core::linalg::BlockSparseArrowheadMatrix& A,
		const open3d::core::Tensor& B
) {
	o3c::Tensor U_diagonal_upper_left, U_sparse_blocks, U_factorized_dense_lower_right;
	std::tie(U_diagonal_upper_left, U_sparse_blocks, U_factorized_dense_lower_right) = FactorizeBlockSparseArrowheadCholesky_Upper(A);
	utility::LogError("Not implemented");
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
FactorizeBlockSparseArrowheadCholesky_Upper(const BlockSparseArrowheadMatrix& A) {
	o3c::Tensor L_diagonal_upper_left;
	FactorizeBlocksCholesky(L_diagonal_upper_left, A.diagonal_blocks.Slice(0, 0, A.arrow_base_block_index), UpLoTriangular::LOWER);
	//__DEBUG
	// L_diagonal_upper_left.Save("/mnt/Data/Reconstruction/output/matrix_experiments/L_diag_blocks_cpu.npy");

	o3c::Tensor L_inv_diagonal_upper_left = InvertTriangularBlocks(L_diagonal_upper_left, UpLoTriangular::LOWER);
	o3c::Tensor U_diagonal_upper_left = L_diagonal_upper_left.Transpose(1, 2);

	//__DEBUG
	o3c::Tensor L_inv_diagonal_upper_left_CPU = L_inv_diagonal_upper_left.To(o3c::Device("CPU:0"));
	// L_inv_diagonal_upper_left_CPU.Save("/mnt/Data/Reconstruction/output/matrix_experiments/L_diag_inv_blocks_cuda.npy");

	o3c::Tensor U_sparse_blocks = core::linalg::MatmulBlockSparseRowWisePadded(L_inv_diagonal_upper_left, A.upper_blocks, A.upper_block_coordinates);

	//__DEBUG
	o3c::Tensor U_sparse_blocks_CPU = U_sparse_blocks.To(o3c::Device("CPU:0"));
	// U_sparse_blocks_CPU.Save("/mnt/Data/Reconstruction/output/matrix_experiments/U_sparse_blocks_cuda.npy");

	o3c::Tensor U_factorized_dense_corner;
	internal::FactorizeBlockSparseCholeskyCorner(U_factorized_dense_corner, U_sparse_blocks, A);

	return std::make_tuple(U_diagonal_upper_left, U_sparse_blocks, U_factorized_dense_corner);
}

namespace internal {

void FactorizeBlockSparseCholeskyCorner(
		open3d::core::Tensor& factorized_upper_dense_corner,
		const open3d::core::Tensor& factorized_upper_blocks,
		const BlockSparseArrowheadMatrix& A
) {
	core::ExecuteOnDevice(
			A.diagonal_blocks.GetDevice(),
			[&]() {
				internal::FactorizeBlockSparseCholeskyCorner<open3d::core::Device::DeviceType::CPU>(
						factorized_upper_dense_corner, factorized_upper_blocks, A
				);
			},
			[&]() {
				NNRT_IF_CUDA(
						internal::FactorizeBlockSparseCholeskyCorner<open3d::core::Device::DeviceType::CUDA>(
								factorized_upper_dense_corner, factorized_upper_blocks, A
						);
				);
			}
	);
}

} // namespace internal


} // namespace nnrt::core::linalg

