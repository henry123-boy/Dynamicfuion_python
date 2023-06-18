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
#include "core/platform_independence/Atomics.h"
#include "core/kernel/MathTypedefs.h"
#include "core/Dispatch.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/BlasWrapper.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

#define ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT 2000


//TODO: need a proper AtomicTensor type, compatible with Open3D tensor, with a method to fill values atomically in a cross-platform fashion
static void RunBlockSumChecks(
		o3c::Tensor& sums, int sum_count, const o3c::Tensor& blocks, const o3c::Tensor& block_sum_indices, int block_count
) {
	// counters & checks
	o3c::Device device = blocks.GetDevice();
	o3c::AssertTensorDtype(sums, o3c::Float32);
	o3c::AssertTensorDtype(blocks, o3c::Float32);
	o3c::AssertTensorDtype(block_sum_indices, o3c::Int32);
	int64_t block_size = blocks.GetShape(1);
	int64_t max_block_count = blocks.GetShape(0);
	int64_t max_sum_count = sums.GetShape(0);
	o3c::AssertTensorShape(sums, { max_sum_count, block_size, block_size });
	o3c::AssertTensorShape(blocks, { max_block_count, block_size, block_size });
	o3c::AssertTensorShape(block_sum_indices, { max_block_count });

	o3c::AssertTensorDevice(sums, device);
	o3c::AssertTensorDevice(block_sum_indices, device);

	if (block_count > max_block_count) {
		utility::LogError("Block count, {}, exceeds allowed maximum, {}.", block_count, max_block_count);
	}
	if (sum_count > max_sum_count) {
		utility::LogError("Sum count, {}, exceeds allowed maximum, {}.", sum_count, max_sum_count);
	}
}

#ifdef __CUDACC__
void ComputeBlockSumsCUDA(o3c::Tensor& sums, int sum_count, const o3c::Tensor& blocks, const o3c::Tensor& block_sum_indices, int block_count) {
	RunBlockSumChecks(sums, sum_count, blocks, block_sum_indices, block_count);
	sums.Fill(0);
	o3c::Device device = blocks.GetDevice();
	int64_t block_size = blocks.GetShape(1);
	int64_t block_stride = block_size * block_size;
	auto sum_data = sums.GetDataPtr<float>();
	auto block_sum_index_data = block_sum_indices.GetDataPtr<int32_t>();
	auto block_data = blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device,
			block_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_block = workload_idx / block_stride;
				int64_t i_coefficient = workload_idx % block_stride;
				int32_t i_sum = block_sum_index_data[i_block];
				atomicAdd(sum_data + i_sum * block_stride + i_coefficient, block_data[workload_idx]);
			}
	);
}

#else

void ComputeBlockSumsCPU(
		o3c::Tensor& sums, std::vector<std::atomic<float>>& sum_blocks_atomic,
		int sum_count, const o3c::Tensor& blocks, const o3c::Tensor& block_sum_indices, int block_count
) {
	RunBlockSumChecks(sums, sum_count, blocks, block_sum_indices, block_count);

	o3c::Device device = blocks.GetDevice();
	int64_t block_size = blocks.GetShape(1);
	int64_t block_stride = block_size * block_size;

	int64_t max_sum_count = sums.GetShape(0);
	if (sum_blocks_atomic.size() != static_cast<unsigned long>(max_sum_count * block_stride)) {
		utility::LogError("Expecting atomics array to have the same length as block tensor element count, but the former is {} and the latter is {}.",
		                  sum_blocks_atomic.size(), max_sum_count * block_stride);
	}

	o3c::ParallelFor(
			device,
			sum_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				sum_blocks_atomic[workload_idx].store(0);
			}
	);

	auto sum_data = sums.GetDataPtr<float>();
	auto block_sum_index_data = block_sum_indices.GetDataPtr<int32_t>();
	auto block_data = blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device,
			block_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_block = workload_idx / block_stride;
				int64_t i_coefficient = workload_idx % block_stride;
				int32_t i_sum = block_sum_index_data[i_block];
				atomicAdd_CPU(sum_blocks_atomic[i_sum * block_stride + i_coefficient], block_data[workload_idx]);
			}
	);

	// copy over data from atomics to tensor on CPU
	o3c::ParallelFor(
			device,
			sum_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				sum_data[workload_idx] = sum_blocks_atomic[workload_idx].load();
			}
	);
}

#endif


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

	auto source_upper_block_data = A.upper_blocks.GetDataPtr<float>();
	auto factorized_upper_block_data = factorized_upper_blocks.GetDataPtr<float>();
	//TODO: return this too
	auto source_breadboard_data = A.upper_block_breadboard.GetDataPtr<int16_t>();


	int64_t dense_corner_size = breadboard_width * block_size;
	factorized_dense_corner_block = o3c::Tensor({dense_corner_size, dense_corner_size}, o3c::Float32, device);
	auto factorized_upper_dense_data = factorized_dense_corner_block.GetDataPtr<float>();


	int64_t block_stride = block_size * block_size;

	// for holding temporary block sums
	o3c::Tensor sums = o3c::Tensor({breadboard_width, block_size, block_size}, o3c::Float32, device);
	auto sum_data = sums.GetDataPtr<float>();

	// for holding temporary block products
	o3c::Tensor products = o3c::Tensor({ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT, block_size, block_size}, o3c::Float32, device);
	auto product_data = products.GetDataPtr<float>();
	auto product_addresses = reinterpret_cast<float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT, device)
	);
	o3c::ParallelFor(
			device,
			ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				product_addresses[workload_idx] = product_data + workload_idx * block_stride;
			}
	);
	auto product_lhs_addresses = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT, device)
	);
	auto product_rhs_addresses = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT, device)
	);
	o3c::Tensor product_sum_indices = o3c::Tensor({ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT}, o3c::Int32, device);
	auto product_sum_index_data = product_sum_indices.GetDataPtr<int32_t>();


#ifndef __CUDACC__
	std::vector<std::atomic<float>> sum_blocks_atomic(breadboard_width * block_stride);
#endif
	// ~breadboard column
	int i = 0;
	for (int i_diagonal_block = A.arrow_base_block_index; i_diagonal_block < diagonal_block_count; i_diagonal_block++, i++) {
		int32_t block_count_in_row_or_column = diagonal_block_count - i_diagonal_block;
		int32_t block_count_above_blocks_in_row_i = i_diagonal_block - 1;


		// determine block addresses for products
		NNRT_DECLARE_ATOMIC(int, product_count_atomic);
		NNRT_INITIALIZE_ATOMIC(int, product_count_atomic, 0);

		o3c::ParallelFor(
				device,
				block_count_in_row_or_column * block_count_above_blocks_in_row_i,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int j = static_cast<int32_t>(workload_idx) / block_count_above_blocks_in_row_i;
					int i_product = NNRT_ATOMIC_ADD(product_count_atomic, 1);
					product_sum_index_data[i_product] = j;
					if (j == 0) {
						// first "block_count_above_blocks_in_row_i" blocks are above the diagonal block
						int64_t k = workload_idx;
						const float* block_data;
						if (k < A.arrow_base_block_index) {
							int16_t i_edge = source_breadboard_data[k * breadboard_width + i];
							if (i_edge == -1) return;
							block_data = factorized_upper_block_data + i_edge * 36;
						} else {
							block_data = factorized_upper_dense_data + (k * block_size * dense_corner_size) + (i * block_size);
						}
						product_lhs_addresses[i_product] = block_data;
						product_rhs_addresses[i_product] = block_data;

					} else {
						auto offset_workload_idx = static_cast<int32_t>(workload_idx - block_count_above_blocks_in_row_i);
						int k = offset_workload_idx % block_count_above_blocks_in_row_i;
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
						product_lhs_addresses[i_product] = block_ki_data;
						product_rhs_addresses[i_product] = block_kj_data;
					}
				}
		);
		int product_count = NNRT_GET_ATOMIC_VALUE_HOST(product_count_atomic); NNRT_CLEAN_UP_ATOMIC(product_count_atomic);
		float alpha = 1.f, beta = 0.f;

#ifdef __CUDACC__
		cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
		NNRT_CUBLAS_CHECK(
				gemm_batched_cuda<float>(
						handle, CUBLAS_OP_T, CUBLAS_OP_N,
						block_size, block_size, block_size,
						&alpha,
						product_lhs_addresses, block_size,
						product_rhs_addresses, block_size,
						&beta,
						product_addresses, block_size,
						product_count
				),
				"cuda batched gemm failed"
		);
#else
		gemm_batched_cpu<float>(CblasRowMajor, CblasTrans, CblasNoTrans,
								block_size, block_size, block_size,
								alpha,
		                        product_lhs_addresses, block_size,
								product_rhs_addresses, block_size,
								beta,
								product_addresses, block_size,
								product_count);
#endif

#ifdef __CUDACC__
		ComputeBlockSumsCUDA(sums, block_count_in_row_or_column, products, product_sum_indices, product_count);
#else
		ComputeBlockSumsCPU(sums, sum_blocks_atomic, block_count_in_row_or_column, products, product_sum_indices, product_count);
#endif

		o3c::Tensor factorized_U_diagonal_block = A.diagonal_blocks.GetItem(o3c::TensorKey::Index(i_diagonal_block)).Clone();
		o3c::Tensor uTu_blocks_above_diagonal_sum = sums.GetItem(o3c::TensorKey::Index(0));
		factorized_U_diagonal_block -= uTu_blocks_above_diagonal_sum;

		//__DEBUG
		// factorized_U_diagonal_block = factorized_U_diagonal_block.Transpose(0, 1); // layout-flip
		auto factorized_U_diagonal_block_data = factorized_U_diagonal_block.GetDataPtr<float>();
		CuSolverContext::GetInstance()->GetHandle();
#ifdef __CUDACC__
		cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();
		NNRT_CUSOLVER_CHECK(
				potrf_cuda<float>(
						cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
						block_size, factorized_U_diagonal_block_data, block_size
				), "Batched portf failed in SolveBlockDiagonalCUDACholesky"
		);
#else
		NNRT_LAPACK_CHECK(
				potrf_cpu<float>(
						LAPACK_COL_MAJOR, 'L', block_size, factorized_U_diagonal_block_data, block_size),
				"potrf failed in SolveBlockDiagonalCholeskyCPU"
		);
#endif
		o3c::Tensor inverted_factorized_L_diagonal_block = factorized_U_diagonal_block.Clone().Transpose(0, 1);
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
		//__DEBUG
		// factorized_U_diagonal_block = factorized_U_diagonal_block.Transpose(0, 1);
		// factorized_U_diagonal_block_data = factorized_U_diagonal_block.GetDataPtr<float>();
		// inverted_factorized_L_diagonal_block_data = inverted_factorized_L_diagonal_block.GetDataPtr<float>();

		auto factorized_upper_dense_row_data = factorized_upper_dense_data + (i * block_size * dense_corner_size);
		o3c::MemoryManager::Memcpy(
				factorized_upper_dense_row_data + (i * block_stride),
				device, factorized_U_diagonal_block_data, device, block_stride * sizeof(float)
		);

		int32_t non_diagonal_blocks_in_row_count = block_count_in_row_or_column - 1; // for clarity
		int64_t j_offset = diagonal_block_count - non_diagonal_blocks_in_row_count;

		// calculate "updated" blocks, i.e. source_block - sum_block, and store them again in sum block array

		auto non_diagonal_sum_data = sum_data + block_stride;

		o3c::ParallelFor(
				device,
				non_diagonal_blocks_in_row_count * block_stride,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int64_t j = j_offset + workload_idx / block_stride;
					int64_t i_coefficient_in_block = workload_idx % block_stride;
					int16_t i_block = source_breadboard_data[i_diagonal_block * breadboard_width + j];
					if (i_block == -1) {
						non_diagonal_sum_data[workload_idx] = -non_diagonal_sum_data[workload_idx];
					} else {
						non_diagonal_sum_data[workload_idx] =
								source_upper_block_data[i_block * block_stride + i_coefficient_in_block] - non_diagonal_sum_data[workload_idx];
					}
				}
		);

		o3c::ParallelFor(
				device,
				non_diagonal_blocks_in_row_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_nondiagonal_block) {
					int64_t j = j_offset + i_nondiagonal_block;
					product_lhs_addresses[i_nondiagonal_block] = inverted_factorized_L_diagonal_block_data;
					product_rhs_addresses[i_nondiagonal_block] = non_diagonal_sum_data + i_nondiagonal_block * block_stride;
					product_addresses[i_nondiagonal_block] = factorized_upper_dense_row_data + (j * block_stride);
				}
		);
		//TODO: use magmablas_strmm_vbatched
	}
	o3c::MemoryManager::Free(product_lhs_addresses, device);
	o3c::MemoryManager::Free(product_rhs_addresses, device);
	o3c::MemoryManager::Free(product_addresses, device);
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
