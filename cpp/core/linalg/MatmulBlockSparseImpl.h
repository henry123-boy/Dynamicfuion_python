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
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>

// local includes
#include "core/linalg/MatmulBlockSparse.h"
#include "core/platform_independence/Qualifiers.h"
#include "LinalgUtils.h"
#include "core/linalg/BlasWrapper.h"
#include "core/platform_independence/Atomics.h"
#include "core/platform_independence/AtomicTensor.h"
#include "BlockSums.h"
#include "TransposeBlocks.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

#define ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT 8000

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates,
		bool padded
) {
	// counters and checks
	o3c::Device device = blocks_b.GetDevice();
	o3c::AssertTensorDtype(blocks_b, o3c::Float32);
	int64_t block_count = blocks_b.GetShape(0);
	int64_t block_size = blocks_b.GetShape(1);
	o3c::AssertTensorShape(blocks_b, { block_count, block_size, block_size });

	o3c::AssertTensorShape(blocks_b_coordinates, { block_count, 2 });
	o3c::AssertTensorDtype(blocks_b_coordinates, o3c::Int32);
	o3c::AssertTensorDevice(blocks_b_coordinates, device);

	int64_t a_block_count = blocks_a.GetShape(0);
	o3c::AssertTensorShape(blocks_a, { a_block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks_a, o3c::Float32);
	o3c::AssertTensorDevice(blocks_a, device);

	// prep inputs
	auto blocks_b_data = blocks_b.GetDataPtr<float>();
	auto blocks_b_coordinate_data = blocks_b_coordinates.GetDataPtr<int32_t>();
	auto blocks_a_data = blocks_a.GetDataPtr<float>();

	// prep outputs
	o3c::Tensor product_blocks = o3c::Tensor(blocks_b.GetShape(), o3c::Float32, device);
	auto blocks_c_data = product_blocks.GetDataPtr<float>();

	// loop over blocks and assign multiplication block triplets
#ifdef __CUDACC__
	const float** a_blocks_device;
	const float** b_blocks_device;
	float** c_blocks_device;
	auto size_of_pointer_array =  block_count * sizeof(float*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&a_blocks_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&b_blocks_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&c_blocks_device, size_of_pointer_array));
#else
	const float* a_blocks_device[block_count];
	const float* b_blocks_device[block_count];
	float* c_blocks_device[block_count];
#endif
	o3c::Tensor padding_block = o3c::Tensor::Zeros({block_size, block_size}, o3c::Float32, device);
	float* padding_data = padding_block.GetDataPtr<float>();


	int64_t block_stride = block_size * block_size;
	o3c::Tensor mask({block_count}, o3c::Bool, device);
	auto mask_data = mask.GetDataPtr<bool>();
	o3c::ParallelFor(
			device,
			block_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_input_block) {
				int i_row = blocks_b_coordinate_data[i_input_block * 2];
				if (i_row < a_block_count) {
					a_blocks_device[i_input_block] = blocks_a_data + i_row * block_stride;
					b_blocks_device[i_input_block] = blocks_b_data + i_input_block * block_stride;
					mask_data[i_input_block] = true;
				} else {
					a_blocks_device[i_input_block] = padding_data;
					b_blocks_device[i_input_block] = padding_data;
					mask_data[i_input_block] = false;
				}
				c_blocks_device[i_input_block] = blocks_c_data + i_input_block * block_stride;
			}
	);
	auto block_size_int32 = static_cast<int32_t>(block_size);
	float alpha = 1, beta = 0;

#ifdef __CUDACC__
	cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
	NNRT_CUBLAS_CHECK(
			// Note: A<->B matrix order flippage is due to difference in layout -- gemm_batched_cuda assumes column-major,
			// while open3d::core::Tensors are row-major
			gemm_batched_cuda<float>(
					handle, CUBLAS_OP_N, CUBLAS_OP_N,
					block_size_int32, block_size_int32, block_size_int32,
					&alpha,
					b_blocks_device, block_size_int32,
					a_blocks_device, block_size_int32,
					&beta,
					c_blocks_device, block_size_int32,
					block_count
			),
			"cuda batched gemm failed"
	);
#else
	gemm_batched_cpu<float>(
			CblasRowMajor, CblasNoTrans, CblasNoTrans,
			block_size_int32, block_size_int32, block_size_int32,
			alpha,
			a_blocks_device, block_size_int32,
			b_blocks_device, block_size_int32,
			beta,
			c_blocks_device, block_size_int32,
			block_count
	);
#endif

#ifdef __CUDACC__
	OPEN3D_CUDA_CHECK(cudaFree(a_blocks_device));
	OPEN3D_CUDA_CHECK(cudaFree(b_blocks_device));
	OPEN3D_CUDA_CHECK(cudaFree(c_blocks_device));
#endif
	if (padded) {
		o3c::TensorKey mask_key = o3c::TensorKey::IndexTensor(mask);
		o3c::Tensor filtered_product_blocks = product_blocks.GetItem(mask_key);
		o3c::Tensor filtered_coordinates = blocks_b_coordinates.GetItem(mask_key);
		return std::make_tuple(filtered_product_blocks, filtered_coordinates);
	} else {
		return std::make_tuple(product_blocks, o3c::Tensor());
	}
}


template<open3d::core::Device::DeviceType TDeviceType, bool TTransposeA, bool TTransposeB>
void
MatmulBlockSparse_Generic(
		open3d::core::Tensor& blocks,
		open3d::core::Tensor& coordinates,
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& a_block_breadboard,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& b_block_breadboard
) {
	// counts & checks
	o3c::Device device = blocks_a.GetDevice();
	int64_t a_block_count = blocks_a.GetShape(0);
	int64_t block_size = blocks_a.GetShape(1);
	int64_t block_stride = block_size * block_size;

	o3c::AssertTensorShape(blocks_a, { a_block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks_a, o3c::Float32);

	int64_t b_block_count = blocks_b.GetShape(0);

	const auto a_block_row_count = static_cast<int>(a_block_breadboard.GetShape(0));
	const auto a_block_column_count = static_cast<int>(a_block_breadboard.GetShape(1));
	o3c::AssertTensorShape(a_block_breadboard, { a_block_row_count, a_block_column_count });
	o3c::AssertTensorDtype(a_block_breadboard, o3c::Int16);
	o3c::AssertTensorDevice(a_block_breadboard, device);

	o3c::AssertTensorShape(blocks_b, { b_block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks_b, o3c::Float32);
	o3c::AssertTensorDevice(blocks_b, device);

	const auto b_block_row_count = static_cast<int>(b_block_breadboard.GetShape(0));
	const auto b_block_column_count = static_cast<int>(b_block_breadboard.GetShape(1));
	o3c::AssertTensorShape(b_block_breadboard, { b_block_row_count, b_block_column_count });
	o3c::AssertTensorDtype(b_block_breadboard, o3c::Int16);
	o3c::AssertTensorDevice(b_block_breadboard, device);

	bool inner_dimensions_match;
	int out_row_count, out_column_count;
	if (TTransposeA) {
		if (TTransposeB) {
			inner_dimensions_match = a_block_row_count == b_block_column_count;
			out_row_count = a_block_column_count;
			out_column_count = b_block_row_count;

		} else {
			inner_dimensions_match = a_block_row_count == b_block_row_count;
			out_row_count = a_block_column_count;
			out_column_count = b_block_column_count;
		}
	} else {
		if (TTransposeB) {
			inner_dimensions_match = a_block_column_count == b_block_column_count;
			out_row_count = a_block_row_count;
			out_column_count = b_block_row_count;
		} else {
			inner_dimensions_match = a_block_column_count == b_block_row_count;
			out_row_count = a_block_row_count;
			out_column_count = b_block_column_count;
		}
	}
	if (!inner_dimensions_match) {
		utility::LogError("Matrix inner dimensions must but do not match.");
	}
	const int out_block_count = out_row_count * out_column_count;

	// output & accessors

	auto a_block_breadboard_data = a_block_breadboard.GetDataPtr<int16_t>();
	auto b_block_breadboard_data = b_block_breadboard.GetDataPtr<int16_t>();
	auto a_block_data = blocks_a.GetDataPtr<float>();
	auto b_block_data = blocks_b.GetDataPtr<float>();

	o3c::Tensor product_sum_indices = o3c::Tensor({ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT}, o3c::Int32, device);
	auto product_sum_index_data = product_sum_indices.GetDataPtr<int32_t>();

	auto product_lhs_addresses = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT, device)
	);
	auto product_rhs_addresses = reinterpret_cast<const float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT, device)
	);
	o3c::Tensor products({ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT, block_size, block_size}, o3c::Float32, device);
	auto product_data = products.GetDataPtr<float>();
	auto product_addresses = reinterpret_cast<float**>(
			o3c::MemoryManager::Malloc(sizeof(float*) * ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT, device)
	);
	o3c::ParallelFor(
			device,
			ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				product_addresses[workload_idx] = product_data + workload_idx * block_stride;
			}
	);

	// blocks = o3c::Tensor::Zeros({out_block_count, block_size, block_size}, o3c::Float32, device);
	AtomicTensor<TDeviceType, float> product_sums_atomic({out_block_count, block_size, block_size}, device);
	coordinates = o3c::Tensor({out_block_count, 2}, o3c::Int32, device);
	auto coordinate_data = coordinates.GetDataPtr<int32_t>();
	// meshgrid
	o3c::ParallelFor(
			device,
			out_block_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const int workload_idx_int = static_cast<int32_t>(workload_idx);
				int i_output_row = workload_idx_int / out_column_count;
				int i_output_column = workload_idx_int % out_column_count;
				coordinate_data[workload_idx * 2] = i_output_row;
				coordinate_data[workload_idx * 2 + 1] = i_output_column;
			}
	);
	o3c::Tensor out_block_mask = o3c::Tensor::Zeros({a_block_row_count * b_block_column_count}, o3c::Bool, device);
	auto out_block_mask_data = out_block_mask.GetDataPtr<bool>();


	// determine block addresses for row_products
	NNRT_DECLARE_ATOMIC(int, product_count_atomic);
	NNRT_INITIALIZE_ATOMIC(int, product_count_atomic, 0);

	const int32_t b_total_stride = b_block_column_count * b_block_row_count;

	int64_t dense_product_count;

	if (TTransposeA) {
		dense_product_count = a_block_column_count * b_block_column_count * b_block_row_count;
	} else {
		dense_product_count = a_block_row_count * b_block_column_count * b_block_row_count;
	}

	o3c::ParallelFor(
			device,
			dense_product_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const int workload_idx_int = static_cast<int32_t>(workload_idx);
				int i_a_block_row, i_a_block_column, i_b_block_row, i_b_block_column, out_i, out_j;
				if (TTransposeA) {
					out_i = i_a_block_column = workload_idx_int / b_total_stride;
					int i_block_in_b = workload_idx_int - i_a_block_column * b_total_stride;
					if (TTransposeB) {
						i_b_block_column = i_block_in_b / b_block_row_count;
						out_j = i_b_block_row = i_block_in_b % b_block_row_count;
						i_a_block_row = i_b_block_column;
					} else {
						i_b_block_row = i_block_in_b / b_block_column_count;
						out_j = i_b_block_column = i_block_in_b% b_block_column_count;
						i_a_block_row = i_b_block_row;
					}
				} else {
					out_i = i_a_block_row = workload_idx_int / b_total_stride;
					int i_block_in_b = workload_idx_int - i_a_block_row * b_total_stride;
					if (TTransposeB) {
						i_b_block_column = i_block_in_b / b_block_row_count;
						out_j = i_b_block_row = i_block_in_b % b_block_row_count;
						i_a_block_column = i_b_block_column;
					} else {
						i_b_block_row = i_block_in_b / b_block_column_count;
						out_j = i_b_block_column = i_block_in_b % b_block_column_count;
						i_a_block_column = i_b_block_row;
					}
				}

				int16_t i_block_a = a_block_breadboard_data[i_a_block_row * a_block_column_count + i_a_block_column];
				if (i_block_a == -1) return;
				int16_t i_block_b = b_block_breadboard_data[i_b_block_row * b_block_column_count + i_b_block_column];
				if (i_block_b == -1) return;
				const float* block_a_data = a_block_data + i_block_a * block_stride;
				const float* block_b_data = b_block_data + i_block_b * block_stride;

				int i_product = NNRT_ATOMIC_ADD(product_count_atomic, 1);
				if (i_product > ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT) {
					printf("Warning: necessary number of block products for block-sparse matrix product row exceeds allowed "
					       "maximum, %d. Solution will be incorrect/inaccurate. Try adjusting the allowed maximum in the code.\n",
					       ESTIMATE_MAX_POSSIBLE_SPARSE_BLOCK_PRODUCT_COUNT);
				} else {
					int i_output_block = out_i * out_column_count + out_j;
					product_sum_index_data[i_product] = i_output_block;
					out_block_mask_data[i_output_block] = true;
					product_lhs_addresses[i_product] = block_a_data;
					product_rhs_addresses[i_product] = block_b_data;
				}
			}
	);
	int product_count = NNRT_GET_ATOMIC_VALUE_HOST(product_count_atomic);NNRT_CLEAN_UP_ATOMIC(product_count_atomic);
	float alpha = 1.f, beta = 0.f;

#ifdef __CUDACC__
	// Note flippage of transposition operations due to column-major ordering in cuBLAS
	const cublasOperation_t operation_a = TTransposeA ? CUBLAS_OP_N : CUBLAS_OP_T;
	const cublasOperation_t operation_b = TTransposeB ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
	NNRT_CUBLAS_CHECK(
			gemm_batched_cuda<float>(
					handle, operation_a, operation_b,
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
	// Note: since products are now transposed due to col-major ordering, their sums will be also transposed
#else
	const CBLAS_TRANSPOSE operation_a = TTransposeA ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE operation_b = TTransposeB ? CblasTrans : CblasNoTrans;
	gemm_batched_cpu<float>(
			CblasRowMajor, operation_a, operation_b,
			block_size, block_size, block_size,
			alpha,
			product_lhs_addresses, block_size,
			product_rhs_addresses, block_size,
			beta,
			product_addresses, block_size,
			product_count
	);
#endif
	internal::ComputeBlockSums(product_sums_atomic, out_block_count, products, product_sum_indices, product_count);


	o3c::MemoryManager::Free(product_lhs_addresses, device);
	o3c::MemoryManager::Free(product_rhs_addresses, device);
	o3c::MemoryManager::Free(product_addresses, device);

	auto mask_key = o3c::TensorKey::IndexTensor(out_block_mask);
	blocks = product_sums_atomic.AsTensor(false).GetItem(mask_key).Contiguous();
#ifdef __CUDACC__
	core::linalg::TransposeBlocksInPlace(blocks);
#endif
	coordinates = coordinates.GetItem(mask_key).Contiguous();
}

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparse(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& a_block_breadboard,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& b_block_breadboard,
		MatrixPreprocessingOperation matrix_b_preprocessing
) {
	o3c::Tensor blocks, coordinates;
	switch (matrix_a_preprocessing) {
		case MatrixPreprocessingOperation::NONE:
			switch (matrix_b_preprocessing) {
				case MatrixPreprocessingOperation::NONE:
					MatmulBlockSparse_Generic<TDeviceType, false, false>(
							blocks, coordinates,
							blocks_a, a_block_breadboard, blocks_b, b_block_breadboard
					);
					break;
				case MatrixPreprocessingOperation::TRANSPOSE:
					MatmulBlockSparse_Generic<TDeviceType, false, true>(
							blocks, coordinates,
							blocks_a, a_block_breadboard, blocks_b, b_block_breadboard
					);
					break;
			}
			break;
		case MatrixPreprocessingOperation::TRANSPOSE:
			switch (matrix_b_preprocessing) {
				case MatrixPreprocessingOperation::NONE:
					MatmulBlockSparse_Generic<TDeviceType, true, false>(
							blocks, coordinates,
							blocks_a, a_block_breadboard, blocks_b, b_block_breadboard
					);
					break;
				case MatrixPreprocessingOperation::TRANSPOSE:
					MatmulBlockSparse_Generic<TDeviceType, true, true>(
							blocks, coordinates,
							blocks_a, a_block_breadboard, blocks_b, b_block_breadboard
					);
					break;
			}
			break;
	}
	return std::make_tuple(blocks, coordinates);
}


} // namespace nnrt::core::linalg::internal