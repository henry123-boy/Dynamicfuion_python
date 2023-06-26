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
#include "core/linalg/BlasAuxiliary.h"
#include "MagmaManager.h"

//__DEBUG
#include "core/functional/Sorting.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

#define ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT 2000
#define ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT 200


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
	if (sum_blocks_atomic.size() < static_cast<unsigned long>(max_sum_count * block_stride)) {
		utility::LogError("Expecting atomics array to have at least the length of the block tensor element count, "
		                  "but the former is {} and the latter is {}.",
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
		open3d::core::Tensor& factorized_dense_corner_matrix,
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

	int64_t A_upper_block_count = A.upper_blocks.GetShape(0);
	o3c::AssertTensorShape(A.upper_blocks, { A_upper_block_count, block_size, block_size });
	o3c::AssertTensorDtype(A.upper_blocks, o3c::Float32);
	o3c::AssertTensorDevice(A.upper_blocks, device);

	o3c::AssertTensorShape(factorized_upper_blocks, { A_upper_block_count, block_size, block_size });
	o3c::AssertTensorDtype(factorized_upper_blocks, o3c::Float32);
	o3c::AssertTensorDevice(factorized_upper_blocks, device);

	auto source_upper_block_data = A.upper_blocks.GetDataPtr<float>();
	auto factorized_upper_block_data = factorized_upper_blocks.GetDataPtr<float>();
	//TODO: return this too
	auto source_breadboard_data = A.upper_block_breadboard.GetDataPtr<int16_t>();

	// int64_t dense_corner_size = breadboard_width * block_size;
	o3c::Tensor factorized_corner_blocks = o3c::Tensor::Zeros({breadboard_width, breadboard_width, block_size, block_size}, o3c::Float32, device);
	auto factorized_corner_block_data = factorized_corner_blocks.GetDataPtr<float>();

	//__DEBUG
	auto row1 = factorized_corner_blocks.Slice(0,0,1);
	auto row2 = factorized_corner_blocks.Slice(0,1,2);

	int64_t block_stride = block_size * block_size;

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
#ifdef __CUDACC__
	// workaround MAGMA's lack of const-correctness in magmablas_strmm_batched
	auto repeated_inv_block_address = reinterpret_cast<float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT, device)
	);
#else
	auto repeated_inv_block_address = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT, device)
	);
#endif
	auto product_rhs_addresses = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT, device)
	);
	auto non_diagonal_block_addresses = reinterpret_cast<float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT, device)
	);
	o3c::Tensor product_sum_indices = o3c::Tensor({ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT}, o3c::Int32, device);
	auto product_sum_index_data = product_sum_indices.GetDataPtr<int32_t>();


#ifndef __CUDACC__
	std::vector<std::atomic<float>> sum_blocks_atomic(breadboard_width * block_stride);
#endif
	// ~breadboard column
	int i_block_row_in_corner = 0;
	for (int i_block_row_in_matrix = A.arrow_base_block_index;
	     i_block_row_in_matrix < diagonal_block_count; i_block_row_in_matrix++, i_block_row_in_corner++) {
		int32_t block_count_in_row_or_column = diagonal_block_count - i_block_row_in_matrix;


		int32_t non_diagonal_blocks_in_row_count = block_count_in_row_or_column - 1; // for clarity

		o3c::Tensor block_row = factorized_corner_blocks
				.Slice(0, i_block_row_in_corner, i_block_row_in_corner + 1)
				.Slice(1, i_block_row_in_corner, factorized_corner_blocks.GetShape(1)).Reshape({-1, block_size, block_size});
		auto block_row_data = block_row.GetDataPtr<float>();


		// determine block addresses for products
		NNRT_DECLARE_ATOMIC(int, product_count_atomic);
		NNRT_INITIALIZE_ATOMIC(int, product_count_atomic, 0);

		//__DEBUG
		int inspected_row = 245;

		o3c::ParallelFor(
				device,
				block_count_in_row_or_column * i_block_row_in_matrix,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int workload_idx_int = static_cast<int32_t>(workload_idx);
					// block row above
					int k_matrix = workload_idx_int / block_count_in_row_or_column;
					// block column in breadboard
					int i_nonzero_block_in_row = workload_idx_int % block_count_in_row_or_column;
					int j_breadboard = i_block_row_in_corner + i_nonzero_block_in_row;
					// first "block_count_above_blocks_in_row_i" blocks are above the diagonal block, i.e. block at j == 0
					if (i_nonzero_block_in_row == 0) {
						const float* block_data;
						if (k_matrix < A.arrow_base_block_index) {
							int16_t i_edge = source_breadboard_data[k_matrix * breadboard_width + i_block_row_in_corner];
							if (i_edge == -1) return;
							block_data = factorized_upper_block_data + i_edge * 36;
						} else {
							int k_corner = k_matrix - A.arrow_base_block_index;
							block_data =
									factorized_corner_block_data +
									(k_corner * breadboard_width + i_block_row_in_corner) * block_stride;
						}
						int i_product = NNRT_ATOMIC_ADD(product_count_atomic, 1);
						if (i_product > ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT) {
							printf("Warning: necessary number of products for factorizing block-sparse arrowhead matrix corner row exceeds allowed "
							       "maximum, %d. Factorization will be inaccurate. Try adjusting the allowed maximum in the code\n",
							       ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT);
						} else {
							product_sum_index_data[i_product] = i_nonzero_block_in_row;
							product_lhs_addresses[i_product] = block_data;
							product_rhs_addresses[i_product] = block_data;
							//__DEBUG
							// if(i_block_row_in_matrix == inspected_row){
							// 	if(k_matrix < A.arrow_base_block_index){
							// 		printf("%i, %i, %i,    %f, %f, %f   %f\n", k_matrix, i_product, source_breadboard_data[k_matrix * breadboard_width + i_block_row_in_corner], block_data[3], block_data[4], block_data[5], block_data[35]);
							// 	}else{
							// 		printf("%i, %i, -1,\n", k_matrix, i_product);
							// 	}
							// }
						}
					} else {
						const float* block_ki_data;
						const float* block_kj_data;
						if (k_matrix < A.arrow_base_block_index) {
							int16_t i_block_ki = source_breadboard_data[k_matrix * breadboard_width + i_block_row_in_corner];
							if (i_block_ki == -1) return;
							int16_t i_block_kj = source_breadboard_data[k_matrix * breadboard_width + j_breadboard];
							if (i_block_kj == -1) return;
							block_ki_data = factorized_upper_block_data + i_block_ki * 36;
							block_kj_data = factorized_upper_block_data + i_block_kj * 36;
						} else {
							int k_corner = k_matrix - A.arrow_base_block_index;
							block_ki_data =
									factorized_corner_block_data + (k_corner * breadboard_width + i_block_row_in_corner) * block_stride;
							block_kj_data = factorized_corner_block_data + (k_corner * breadboard_width + j_breadboard) * block_stride;
						}
						int i_product = NNRT_ATOMIC_ADD(product_count_atomic, 1);
						if (i_product > ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT) {
							printf("Warning: necessary number of products for factorizing block-sparse arrowhead matrix corner row exceeds allowed "
							       "maximum, %d. Factorization will be inaccurate. Try adjusting the allowed maximum in the code.\n",
							       ESTIMATE_MAX_POSSIBLE_CHOLESKY_BLOCK_ROW_PRODUCT_COUNT);
						} else {
							product_sum_index_data[i_product] = i_nonzero_block_in_row;
							product_lhs_addresses[i_product] = block_ki_data;
							product_rhs_addresses[i_product] = block_kj_data;
						}
					}
				}
		);
		int product_count = NNRT_GET_ATOMIC_VALUE_HOST(product_count_atomic);NNRT_CLEAN_UP_ATOMIC(product_count_atomic);
		float alpha = 1.f, beta = 0.f;

#ifdef __CUDACC__
		cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
		NNRT_CUBLAS_CHECK(
				gemm_batched_cuda<float>(
						handle, CUBLAS_OP_N, CUBLAS_OP_T,
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
		//Note: since products were transposed for CUDA version, output non-diagonal blocks are now transposed
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

		// __DEBUG
		if(i_block_row_in_matrix == inspected_row) {
			auto psi = product_sum_indices.Slice(0, 0, product_count).To(o3c::Device("CPU:0"));
			auto P = products.Slice(0, 0, product_count).To(o3c::Device("CPU:0"));
			auto index = core::functional::ArgSortTensorAlongLastDimension(psi, false);
			psi = psi.GetItem(o3c::TensorKey::IndexTensor(index));
			P = P.GetItem(o3c::TensorKey::IndexTensor(index));
			o3c::Tensor U_Z = block_row.GetItem(o3c::TensorKey::Index(1)).To(o3c::Device("CPU:0"));
			printf("zero\n");
		}

#ifdef __CUDACC__
		ComputeBlockSumsCUDA(block_row, block_count_in_row_or_column, products, product_sum_indices, product_count);
#else
		ComputeBlockSumsCPU(block_row, sum_blocks_atomic, block_count_in_row_or_column, products, product_sum_indices, product_count);
#endif

		//__DEBUG
		if(i_block_row_in_matrix == inspected_row) {
			o3c::Tensor U_sum = block_row.GetItem(o3c::TensorKey::Index(1)).To(o3c::Device("CPU:0"));
			printf("sum\n");
		}

		o3c::Tensor factorized_UT_diagonal_block = A.diagonal_blocks.GetItem(o3c::TensorKey::Index(i_block_row_in_matrix)).Clone();
		o3c::Tensor uTu_blocks_above_diagonal_sum = block_row.GetItem(o3c::TensorKey::Index(0));
		factorized_UT_diagonal_block -= uTu_blocks_above_diagonal_sum;

		auto factorized_UT_diagonal_block_data = factorized_UT_diagonal_block.GetDataPtr<float>();
#ifdef __CUDACC__
		cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();
		NNRT_CUSOLVER_CHECK(
				potrf_cuda<float>(
						// cuSOLVER uses col-major order, therefore this will produce lower-triangular (LT) result here for row-major ordering
						cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
						block_size, factorized_UT_diagonal_block_data, block_size
				), "Batched portf failed in SolveBlockDiagonalCUDACholesky"
		);
#else
		NNRT_LAPACK_CHECK(
				potrf_cpu<float>(
						LAPACK_ROW_MAJOR, 'U', block_size, factorized_UT_diagonal_block_data, block_size),
				"potrf failed in SolveBlockDiagonalCholeskyCPU"
		);
#endif

		o3c::Tensor inverted_factorized_LT_diagonal_block = factorized_UT_diagonal_block.Contiguous().Transpose(0, 1);
		auto inverted_factorized_LT_diagonal_block_data = inverted_factorized_LT_diagonal_block.GetDataPtr<float>();

		//__DEBUG
		if(i_block_row_in_matrix == inspected_row) {
			auto factorized_U_diagonal_block_CPU = factorized_UT_diagonal_block.To(o3c::Device("CPU:0"));
			printf("hi\n");
		}

#ifdef __CUDACC__
		// for CUDA, copy the transposed version instead, since the non-transposed one is in col-major order
		o3c::MemoryManager::Memcpy(
				block_row_data, device, inverted_factorized_LT_diagonal_block_data, device, block_stride * sizeof(float)
		);
#else
		o3c::MemoryManager::Memcpy(
				block_row_data, device, factorized_UT_diagonal_block_data, device, block_stride * sizeof(float)
		);
#endif

		// invert diagonal lower-triangular block
#ifdef __CUDACC__
		// column-major ordering here
		trtri_cuda(cusolver_dn_handle, block_size, inverted_factorized_LT_diagonal_block_data, block_size, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
				   cublasDiagType_t::CUBLAS_DIAG_NON_UNIT);
#else
		NNRT_LAPACK_CHECK(
				trtri_cpu<float>(
						// triangular upper/lower variant "flips" because matrix layout also flips (row->col major) during the computation
						LAPACK_COL_MAJOR, 'U', 'N', block_size, inverted_factorized_LT_diagonal_block_data, block_size
				),
				"trtri failed in FactorizeBlockSparseCholeskyCorner_TypeDispatched"
		);
#endif
		//__DEBUG
		if(i_block_row_in_matrix == inspected_row) {
			auto inverted_factorized_L_diagonal_block_CPU = inverted_factorized_LT_diagonal_block.To(o3c::Device("CPU:0"));
			printf("hi\n");
		}

		// calculate "updated" blocks, i.e. source_block - sum_block, and store them again in sum block array
		auto non_diagonal_sum_data = block_row_data + block_stride;
		int64_t j_non_diagonal_breadboard_offset = breadboard_width - non_diagonal_blocks_in_row_count;
		o3c::ParallelFor(
				device,
				non_diagonal_blocks_in_row_count * block_stride,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int64_t j_block_breadboard = j_non_diagonal_breadboard_offset + workload_idx / block_stride;
					int64_t i_coefficient_in_block = workload_idx % block_stride;
					int16_t i_block = source_breadboard_data[i_block_row_in_matrix * breadboard_width + j_block_breadboard];
					if (i_block == -1) {
						non_diagonal_sum_data[workload_idx] = -non_diagonal_sum_data[workload_idx];
					} else {
						non_diagonal_sum_data[workload_idx] =
								source_upper_block_data[i_block * block_stride + i_coefficient_in_block] - non_diagonal_sum_data[workload_idx];
					}
				}
		);

		if (non_diagonal_blocks_in_row_count > ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT) {
			utility::LogError(
					"Count of non-diagonal blocks in row, {}, exceeds allowed estimate, {}. Try adjusting the allowed estimate in the code.",
					non_diagonal_blocks_in_row_count, ESTIMATE_MAX_CORNER_ROW_NON_DIAGONAL_BLOCK_COUNT
			);
		}


		o3c::ParallelFor(
				device,
				non_diagonal_blocks_in_row_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_nondiagonal_block) {
					repeated_inv_block_address[i_nondiagonal_block] = inverted_factorized_LT_diagonal_block_data;
					non_diagonal_block_addresses[i_nondiagonal_block] = non_diagonal_sum_data + i_nondiagonal_block * block_stride;
				}
		);

		//__DEBUG
		if(i_block_row_in_matrix == inspected_row) {
			o3c::Tensor L_inv = inverted_factorized_LT_diagonal_block.To(o3c::Device("CPU:0"));
			o3c::Tensor H_prime = block_row.GetItem(o3c::TensorKey::Index(1)).To(o3c::Device("CPU:0"));
			printf("hi\n");
		}

#ifdef __CUDACC__
		MagmaManager::GetInstance().SetDevice(device.GetID());
		trmm_batched_cuda_inplace(magma_side_t::MagmaLeft, magma_uplo_t::MagmaLower,
								  magma_trans_t::MagmaNoTrans, magma_diag_t::MagmaNonUnit,
								  block_size, block_size,
								  alpha,
								  repeated_inv_block_address, block_size,
								  non_diagonal_block_addresses, block_size,
								  non_diagonal_blocks_in_row_count,
								  MagmaManager::GetInstance().GetDefaultQueue());
		//TODO instead of appearing here, can be optimized out into the final block tiling (see end of impl.)
		block_row.Slice(0, 1, -1) = block_row.Slice(0, 1, -1).Transpose(1, 2);
#else
		trmm_batched_cpu_inplace<float>(CblasRowMajor, CBLAS_SIDE::CblasLeft, CblasLower,
		                                CblasNoTrans, CBLAS_DIAG::CblasNonUnit,
		                                block_size, block_size,
		                                alpha,
		                                repeated_inv_block_address, block_size,
		                                non_diagonal_block_addresses, block_size,
		                                non_diagonal_blocks_in_row_count);
#endif
		if(i_block_row_in_matrix == inspected_row) {
			o3c::Tensor UD = block_row.GetItem(o3c::TensorKey::Index(0)).To(o3c::Device("CPU:0"));
			o3c::Tensor U = block_row.GetItem(o3c::TensorKey::Index(1)).To(o3c::Device("CPU:0"));
			printf("bye\n");
		}
	}
	o3c::MemoryManager::Free(product_lhs_addresses, device);
	o3c::MemoryManager::Free(product_rhs_addresses, device);
	o3c::MemoryManager::Free(repeated_inv_block_address, device);
	o3c::MemoryManager::Free(product_addresses, device);
	o3c::MemoryManager::Free(non_diagonal_block_addresses, device);

	int64_t corner_matrix_size = breadboard_width * block_size;
	factorized_dense_corner_matrix = o3c::Tensor({corner_matrix_size, corner_matrix_size}, o3c::Float32, device);
	auto factorized_dense_corner_matrix_data = factorized_dense_corner_matrix.GetDataPtr<float>();

	//TODO: potential speedups: 1) investigate whether creative use of lda or ldda would allow us to forgo this step altogether
	// 2) if (1) is not possible, this can be potentially optimized by not using a temporary 'factorized_corner_blocks'
	// instead using shared memory on GPU as temporary storage for blocks and then just copying each block row to the correct memory location
	// (enabling use of reshape(corner_matrix_size,corner_matrix_size) on the factorized_dense_corner_matrix Tensor)
	o3c::ParallelFor(
			device,
			factorized_corner_blocks.NumElements(),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_element) {
				int64_t i_block = i_block_element / block_stride;
				int64_t i_block_row = i_block / breadboard_width;
				int64_t i_block_column = i_block % breadboard_width;
				int64_t i_element_within_block = i_block_element % block_stride;
				int64_t i_matrix_row = i_block_row * block_size + i_element_within_block / block_size;
				int64_t i_matrix_column = i_block_column * block_size + i_element_within_block % block_size;
				factorized_dense_corner_matrix_data[i_matrix_row * corner_matrix_size + i_matrix_column] =
						factorized_corner_block_data[i_block_element];
			}
	);
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
