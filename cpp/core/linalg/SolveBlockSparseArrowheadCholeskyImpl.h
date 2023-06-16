//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/13/23.
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
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>

// local includes
#include "core/linalg/SolveBlockSparseArrowheadCholesky.h"
#include "core/functional/ParallelPrefixScan.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/Dispatch.h"
#include "core/linalg/LinalgUtils.h"
#include "LapackWrapper.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {


template<typename TMatrix, open3d::core::Device::DeviceType TDeviceType>
void FactorizeBlockSparseCholeskyCorner_TypeDispatched(
		open3d::core::Tensor& factorized_dense_corner_block,
		const open3d::core::Tensor& factorized_upper_blocks,
		const BlockSparseArrowheadMatrix& A
) {
	auto diagonal_block_count = static_cast<int32_t>(A.diagonal_blocks.GetShape(0));
	auto device = A.diagonal_blocks.GetDevice();
	int64_t block_size = A.diagonal_blocks.GetShape(1);

	o3c::AssertTensorShape(A.diagonal_blocks, { diagonal_block_count, block_size, block_size });
	o3c::AssertTensorDtype(A.diagonal_blocks, o3c::Float32);


	int breadboard_width = diagonal_block_count - A.arrow_base_block_index;
	o3c::AssertTensorShape(A.upper_block_breadboard, { diagonal_block_count, breadboard_width });
	o3c::AssertTensorDtype(A.upper_block_breadboard, o3c::Int16);
	o3c::AssertTensorDevice(A.upper_block_breadboard, device);

	int64_t upper_block_count = A.upper_blocks.GetShape(0);
	o3c::AssertTensorShape(A.upper_blocks, { upper_block_count, block_size, block_size });
	o3c::AssertTensorDtype(A.upper_blocks, o3c::Float32);
	o3c::AssertTensorDevice(A.upper_blocks, device);

	o3c::AssertTensorShape(factorized_upper_blocks, { upper_block_count, block_size, block_size });
	o3c::AssertTensorDtype(factorized_upper_blocks, o3c::Float32);
	o3c::AssertTensorDevice(factorized_upper_blocks, device);

	o3c::AssertTensorShape(A.upper_block_coordinates, { upper_block_count, 2 });
	o3c::AssertTensorDtype(A.upper_block_coordinates, o3c::Int32);
	o3c::AssertTensorDevice(A.upper_block_coordinates, device);

	auto source_upper_block_data = A.upper_blocks.GetDataPtr<float>();
	auto factorized_upper_block_data = factorized_upper_blocks.GetDataPtr<float>();
	// breadboard will be updated with non-zero blocks in the corner region
	auto factorized_breadboard = A.upper_block_breadboard.Clone();
	//TODO: return this too
	auto source_breadboard_data = A.upper_block_breadboard.GetDataPtr<int16_t>();
	auto factorized_breadboard_data = factorized_breadboard.GetDataPtr<int16_t>();


	int64_t dense_corner_size = breadboard_width * block_size;
	factorized_dense_corner_block = o3c::Tensor({dense_corner_size, dense_corner_size}, o3c::Float32, device);
	auto factorized_upper_dense_data = factorized_dense_corner_block.GetDataPtr<float>();

	o3c::Tensor sums = o3c::Tensor({breadboard_width, block_size, block_size}, o3c::Float32, device);
	auto sum_data = sums.GetDataPtr<float>();

	int64_t block_stride = block_size * block_size;

#ifndef __CUDACC__
	std::vector<std::atomic<float>> sum_blocks_atomic(breadboard_width * block_stride);
#endif
	// ~breadboard column
	int i = 0;
	for (int i_diagonal_block = A.arrow_base_block_index; i_diagonal_block < diagonal_block_count; i_diagonal_block++, i++) {
		sums.Fill(0);
		int32_t block_count_in_row_or_column = diagonal_block_count - i_diagonal_block;
		int32_t i_start_breadboard_column = breadboard_width - block_count_in_row_or_column;
		int32_t block_count_above_blocks_in_row_i = i_diagonal_block - 1;
		int32_t non_diagonal_blocks_in_row_count = block_count_in_row_or_column - 1;
		int32_t block_count_to_check_for_non_diagonal = non_diagonal_blocks_in_row_count * block_count_above_blocks_in_row_i;

		// compute sums
		o3c::ParallelFor(
				device,
				block_count_above_blocks_in_row_i + block_count_to_check_for_non_diagonal,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					if (workload_idx < block_count_above_blocks_in_row_i) {
						int64_t k = workload_idx;
						const float* block_data;
						if (k < A.arrow_base_block_index) {
							int16_t i_edge = source_breadboard_data[k * breadboard_width + i];
							if (i_edge == -1) return;
							block_data = factorized_upper_block_data + i_edge * 36;
						} else {
							block_data = factorized_upper_dense_data + (k * block_size * dense_corner_size) + (i * block_size);
						}
						Eigen::Map<const TMatrix> factorized_block(block_data);
						auto product = factorized_block.transpose() * factorized_block;
						for (int i_coefficient = 0; i_coefficient < block_stride; i_coefficient++) {
#ifdef __CUDACC__
							atomicAdd(sum_data + i_coefficient, product.coeff(i_coefficient));
#else
							atomicAdd_CPU(sum_blocks_atomic[i_coefficient], product.coeff(i_coefficient));
#endif
						}

					} else {
						auto offset_workload_idx = static_cast<int32_t>(workload_idx - block_count_above_blocks_in_row_i);
						int k = offset_workload_idx % block_count_above_blocks_in_row_i;
						int j = i_start_breadboard_column + offset_workload_idx / block_count_above_blocks_in_row_i;
						const float* block_ki_data;
						const float* block_kj_data;
						if (k < A.arrow_base_block_index) {
							int16_t i_block_ki = source_breadboard_data[k * breadboard_width + i];
							if (i_block_ki == -1) return;
							int16_t i_block_kj = source_breadboard_data[k * breadboard_width + j];
							if (i_block_kj == -1) return;
							block_ki_data = factorized_upper_block_data + i_block_ki * 36;
							block_kj_data = factorized_upper_block_data + i_block_kj * 36;
						} else {
							block_ki_data = factorized_upper_dense_data + (k * block_size * dense_corner_size) + (i * block_size);
							block_kj_data = factorized_upper_dense_data + (k * block_size * dense_corner_size) + (j * block_size);
						}
						Eigen::Map<const TMatrix> factorized_block_ki(block_ki_data);
						Eigen::Map<const TMatrix> factorized_block_kj(block_kj_data);
						auto product = factorized_block_ki.transpose() * factorized_block_kj;

#ifdef __CUDACC__
						auto block_sum_data = sum_data + j * block_stride;
#endif
						for (int i_coefficient = 0; i_coefficient < block_stride; i_coefficient++) {
#ifdef __CUDACC__
							atomicAdd(block_sum_data + i_coefficient, product.coeff(i_coefficient));
#else
							atomicAdd_CPU(sum_blocks_atomic[j * block_stride + i_coefficient], product.coeff(i_coefficient));
#endif
						}
					}
				}
		);
#ifndef __CUDACC__
		// copy over data from atomics to tensor on CPU
		o3c::ParallelFor(
				device,
				block_count_in_row_or_column * block_stride,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					sum_data[workload_idx] = sum_blocks_atomic[workload_idx].load();
				}
		);
#endif
		o3c::Tensor factorized_U_diagonal_block = A.diagonal_blocks.GetItem(o3c::TensorKey::Index(i_diagonal_block)).Clone();
		o3c::Tensor uTu_blocks_above_diagonal_sum = sums.GetItem(o3c::TensorKey::Index(0));
		factorized_U_diagonal_block -= uTu_blocks_above_diagonal_sum;

		factorized_U_diagonal_block = factorized_U_diagonal_block.Transpose(0, 1); // layout-flip
		auto factorized_U_diagonal_block_data = factorized_U_diagonal_block.GetDataPtr<float>();
		CuSolverContext::GetInstance()->GetHandle();
#ifdef __CUDACC__
		cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();
		NNRT_CUSOLVER_CHECK(
				potrf_cuda<float>(
						cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
						block_size, factorized_diagonal_block_data, block_size
				), "Batched portf failed in SolveBlockDiagonalCUDACholesky"
		);
#else
		NNRT_LAPACK_CHECK(
				potrf_cpu<float>(
						LAPACK_COL_MAJOR, 'L', block_size, factorized_U_diagonal_block_data, block_size),
				"potrf failed in SolveBlockDiagonalCholeskyCPU"
		);
#endif
		o3c::Tensor inverted_factorized_L_diagonal_block = factorized_U_diagonal_block.Clone();
		auto inverted_factorized_L_diagonal_block_data = inverted_factorized_L_diagonal_block.GetDataPtr<float>();

#ifdef __CUDACC__
		magma_int_t info;
		trtri_cuda(magma_uplo_t::MagmaUpper, magma_diag_t::MagmaNonUnit, block_size, inverted_factorized_L_diagonal_block_data, block_size, &info);
#else
		NNRT_LAPACK_CHECK(
				trtri_cpu<float>(
						// triangular upper/lower variant "flips" because matrix layout also flips (row->col major) during the computation
						LAPACK_COL_MAJOR, 'U', 'N', block_size, inverted_factorized_L_diagonal_block_data, block_size
				),
				"trtri failed in FactorizeBlockSparseCholeskyCorner_TypeDispatched"
		);
#endif
		factorized_U_diagonal_block = factorized_U_diagonal_block.Transpose(0, 1);
		factorized_U_diagonal_block_data = factorized_U_diagonal_block.GetDataPtr<float>();
		inverted_factorized_L_diagonal_block = inverted_factorized_L_diagonal_block.Transpose(0, 1);
		inverted_factorized_L_diagonal_block_data = inverted_factorized_L_diagonal_block.GetDataPtr<float>();

		auto factorized_upper_dense_row_data = factorized_upper_dense_data + (i * block_size * dense_corner_size);
		o3c::MemoryManager::Memcpy(
				factorized_upper_dense_row_data + (i * block_stride),
				device, factorized_U_diagonal_block_data, device, block_stride * sizeof(float)
		);

		o3c::ParallelFor(
				device,
				non_diagonal_blocks_in_row_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t j) {
					Eigen::Map<TMatrix> sum_block(sum_data + (j + 1) * block_stride);
					int16_t i_block = source_breadboard_data[i_diagonal_block * breadboard_width + i];
					TMatrix source_block;
					if (i_block == -1) {
						source_block = TMatrix::Zero();
					} else {
						source_block = source_upper_block_data[i_block * block_size];
					}
					Eigen::Map<TMatrix> inv_L_diagonal(inverted_factorized_L_diagonal_block_data);
					Eigen::Map<TMatrix> factorized_target_block(factorized_upper_dense_row_data + (j * block_stride));
					factorized_target_block = inv_L_diagonal * (source_block - sum_block);
				}
		);
	}
}

template<open3d::core::Device::DeviceType TDeviceType>
void FactorizeBlockSparseCholeskyCorner(
		open3d::core::Tensor& factorized_upper_dense_corner,
		const open3d::core::Tensor& factorized_blocks_upper,
		const BlockSparseArrowheadMatrix& A
) {
	int64_t block_size = A.diagonal_blocks.GetShape(1);
	DISPATCH_MATRIX_BLOCK_SIZE_TO_EIGEN_TYPE(
			block_size,
			[&]() {
				FactorizeBlockSparseCholeskyCorner_TypeDispatched<matrix_t, TDeviceType>(factorized_upper_dense_corner, factorized_blocks_upper, A);
			}
	);
}

} // namespace nnrt::core::linalg::internal
