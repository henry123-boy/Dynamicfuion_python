//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/25/23.
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
#ifdef BUILD_CUDA_MODULE
// stdlib includes

// third-party includes
#include <open3d/core/Dtype.h>
#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h>

// local includes
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/SolveBlockDiagonalCholesky.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/PointerAggregationForBatchOperations.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<typename scalar_t>
inline void SolveCholeskyBlockDiagonalCUDA_Generic(
		void* A_blocks_data,
		void* B_data,
		const int64_t A_and_B_block_row_count,
		const int64_t B_column_count,
		const int64_t block_count
) {
	scalar_t* A_array[block_count];
	scalar_t* B_array[block_count];
	GetMatrixPointersFromContiguousArrayOfMatrices_AB(A_array, B_array, A_blocks_data, B_data, A_and_B_block_row_count, B_column_count, block_count);
	scalar_t** A_array_device;
	scalar_t** B_array_device;

	auto size_of_pointer_array = block_count * sizeof(scalar_t*);

	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&B_array_device, size_of_pointer_array));

	OPEN3D_CUDA_CHECK(cudaMemcpy(A_array_device, A_array, size_of_pointer_array, cudaMemcpyHostToDevice));
	OPEN3D_CUDA_CHECK(cudaMemcpy(B_array_device, B_array, size_of_pointer_array, cudaMemcpyHostToDevice));

	cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();

	int* info_array;
	OPEN3D_CUDA_CHECK(cudaMalloc(&info_array, block_count * sizeof(int)));

	NNRT_CUSOLVER_CHECK(
			potrf_batched_cuda<scalar_t>(cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, A_and_B_block_row_count, A_array_device,
			                             A_and_B_block_row_count, info_array, block_count), "Batched portf failed in SolveCholeskyBlockDiagonalCUDA"
	);

	cublasHandle_t cublas_handle = CuBLASContext::GetInstance()->GetHandle();
	auto alpha = static_cast<scalar_t>(1);
	NNRT_CUBLAS_CHECK(
			trsm_batched_cuda<scalar_t>(cublas_handle, cublasSideMode_t::CUBLAS_SIDE_LEFT, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
			                            cublasOperation_t::CUBLAS_OP_T, cublasDiagType_t::CUBLAS_DIAG_NON_UNIT, A_and_B_block_row_count,
			                            B_column_count, &alpha, A_array_device, A_and_B_block_row_count, B_array_device,
			                            A_and_B_block_row_count, block_count),
			"Batched trsm failed in SolveCholeskyBlockDiagonalCUDA"
	);
	NNRT_CUBLAS_CHECK(
			trsm_batched_cuda<scalar_t>(cublas_handle, cublasSideMode_t::CUBLAS_SIDE_LEFT, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
			                            cublasOperation_t::CUBLAS_OP_N, cublasDiagType_t::CUBLAS_DIAG_NON_UNIT, A_and_B_block_row_count,
			                            B_column_count, &alpha, A_array_device, A_and_B_block_row_count, B_array_device,
			                            A_and_B_block_row_count, block_count),
			"Batched trsm failed in SolveCholeskyBlockDiagonalCUDA"
	);
	OPEN3D_CUDA_CHECK(cudaFree(info_array));
	OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
	OPEN3D_CUDA_CHECK(cudaFree(B_array_device));
}

void SolveCholeskyBlockDiagonalCUDA(
		void* A_blocks_data,
		void* B_data,
		const int64_t A_and_B_block_row_count,
		const int64_t B_column_count,
		const int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		SolveCholeskyBlockDiagonalCUDA_Generic<scalar_t>(A_blocks_data, B_data, A_and_B_block_row_count,
		                                                 B_column_count, block_count);
	});
}


} // nnrt::core::linalg::internal
#endif