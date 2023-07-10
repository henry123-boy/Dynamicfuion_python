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
#include "core/DeviceSelection.h"
#include "core/linalg/FactorizeBlocksCholesky.h"
#include "core/linalg/InvertBlocks.h"
#include "core/linalg/MatmulBlockSparse.h"
#include "core/linalg/SchurComplement.h"
#include "core/linalg/SolveCholesky.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg {

void SolveBlockSparseArrowheadCholesky(
		open3d::core::Tensor& x,
		const nnrt::core::linalg::BlockSparseArrowheadMatrix& a,
		const open3d::core::Tensor& b
) {

	int64_t b_element_count = b.GetShape(0);
	auto device = a.GetDevice();

	o3c::AssertTensorShape(b, { b_element_count });
	o3c::AssertTensorDevice(b, device);
	o3c::AssertTensorDtype(b, o3c::Float32);

	if (a.diagonal_block_count * a.GetBlockSize() != b_element_count) {
		utility::LogError("Size of lhs tensor b must be \\{{}\\}, matching the leading dimension of rhs matrix a, but is {} instead.",
		                  a.diagonal_block_count * a.GetBlockSize(), b_element_count);
	}

	// compute D^(-1)
	o3c::Tensor inverted_stem = InvertPositiveSemidefiniteBlocks(a.StemDiagonalBlocks());

	// compute D^(-1)B
	o3c::Tensor inverted_stem_and_upper_wing_product_blocks =
			core::linalg::MatmulBlockSparseRowWisePadded(inverted_stem, a.upper_wing_blocks, a.upper_wing_block_coordinates);

	// compute A/D = C - B^(T)D^(-1)B
	o3c::Tensor stem_schur_complement = core::linalg::ComputeSchurComplementOfArrowStem_Dense(a, inverted_stem_and_upper_wing_product_blocks);

	//separate b_D and b_C out from b
	o3c::Tensor b_stem = b.Slice(0, 0, a.arrow_base_block_index * a.GetBlockSize());
	o3c::Tensor b_corner = b.Slice(0, a.arrow_base_block_index * a.GetBlockSize(), a.diagonal_block_count);

	// compute B^(T)D^(-1)b_D = (D^(-1)B)^(T)b_D
	o3c::Tensor b_corner_update = core::linalg::BlockSparseAndVectorProduct(inverted_stem_and_upper_wing_product_blocks,
	                                                                        a.diagonal_block_count - a.arrow_base_block_index,
	                                                                        a.upper_wing_block_coordinates,
	                                                                        MatrixPreprocessingOperation::TRANSPOSE, b_stem);
	// compute b_C - B^(T)D^(-1)b_D
	o3c::Tensor b_corner_prime = b_corner - b_corner_update;

	// solve dense system of equations
	//TODO: add reference-based signature to SolveCholesky that avoids initialization, in order to
	// then init the entire x vector right away and use Slice(...) to reference x_corner and x_stem
	o3c::Tensor x_corner = core::linalg::SolveCholesky(stem_schur_complement, b_corner_prime);

	// compute Bx_C
	o3c::Tensor wing_and_x_corner_product = core::linalg::BlockSparseAndVectorProduct(a.upper_wing_blocks,
	                                                                                  a.arrow_base_block_index,
	                                                                                  a.upper_wing_block_coordinates,
	                                                                                  MatrixPreprocessingOperation::NONE, x_corner);
	o3c::Tensor lhs_stem_operand = b_stem - wing_and_x_corner_product;
	// TODO:
	// auto x_stem =

}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
FactorizeBlockSparseArrowheadCholesky_Upper(const BlockSparseArrowheadMatrix& A) {
	o3c::Tensor L_diagonal_upper_left;
	FactorizeBlocksCholesky(L_diagonal_upper_left, A.StemDiagonalBlocks(), UpLoTriangular::LOWER);

	o3c::Tensor L_inv_diagonal_upper_left = InvertTriangularBlocks(L_diagonal_upper_left, UpLoTriangular::LOWER);
	o3c::Tensor U_diagonal_upper_left = L_diagonal_upper_left.Transpose(1, 2);

	o3c::Tensor U_sparse_blocks = core::linalg::MatmulBlockSparseRowWisePadded(L_inv_diagonal_upper_left, A.upper_wing_blocks,
	                                                                           A.upper_wing_block_coordinates);

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
			A.GetDevice(),
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

