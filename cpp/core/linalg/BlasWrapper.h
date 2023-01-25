//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/12/22.
//  Copyright (c) 2022 Gregory Kramida
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
// third-party
#include <open3d/utility/Logging.h>

// local
#include "core/linalg/LinalgHeadersCUDA.h"
#include "core/linalg/LinalgHeadersCPU.h"
// contains some blas operations currently missing from open3D BlasWrapper.h,
// i.e.
// 1. Batched gemm (general matrix multiplication),
// 2. ?trsm (triangular matrix forward-substituion or back-substituion linear equation system solver)

namespace nnrt::core {


// region ============================= Batched gemm ==================================================================
template<typename scalar_t>
inline void get_matrix_pointers_from_contiguous_array_of_matrices(
		const scalar_t* A_array[], const scalar_t* B_array[], scalar_t* C_array[],
		const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
		int64_t batch_size
) {
	auto A_data = static_cast<const scalar_t*>(A);
	auto B_data = static_cast<const scalar_t*>(B);
	auto C_data = static_cast<scalar_t*>(C);

	auto matrix_A_coefficient_count = m * k;
	auto matrix_B_coefficient_count = k * n;
	auto matrix_C_coefficient_count = m * n;

	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * matrix_A_coefficient_count;
		B_array[i_matrix] = B_data + i_matrix * matrix_B_coefficient_count;
		C_array[i_matrix] = C_data + i_matrix * matrix_C_coefficient_count;
	}
}


template<typename scalar_t>
inline void gemm_batched_cpu(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_TRANSPOSE transpose_B,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const NNRT_CPU_LINALG_INT k,
		const scalar_t alpha,
		const scalar_t* A_array[],
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		const scalar_t* B_array[],
		const NNRT_CPU_LINALG_INT B_leading_dimension,
		const scalar_t beta,
		scalar_t* C_array[],
		const NNRT_CPU_LINALG_INT C_leading_dimension,
		const NNRT_CPU_LINALG_INT batch_size
) {
	open3d::utility::LogError("Unsupported data type.");
}

template<>
inline void gemm_batched_cpu<float>(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_TRANSPOSE transpose_B,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const NNRT_CPU_LINALG_INT k,
		const float alpha,
		const float* A_array[],
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		const float* B_array[],
		const NNRT_CPU_LINALG_INT B_leading_dimension,
		const float beta,
		float* C_array[],
		const NNRT_CPU_LINALG_INT C_leading_dimension,
		const NNRT_CPU_LINALG_INT batch_size
) {
	//TODO: emulate parallelized batched BLAS routine via openMP for loop and regular gemm
#ifdef USE_BLAS
	open3d::utility::LogError("Not currently supported with usage of USE_BLAS (OpenBLAS + LAPACKE).");
#else

	cblas_sgemm_batch(layout, &transpose_A, &transpose_B, &m, &n, &k, &alpha, A_array, &A_leading_dimension, B_array,
	                  &B_leading_dimension, &beta, C_array, &C_leading_dimension, 1, &batch_size);
#endif
}

template<>
inline void gemm_batched_cpu<double>(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_TRANSPOSE transpose_B,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const NNRT_CPU_LINALG_INT k,
		const double alpha,
		const double* A_array[],
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		const double* B_array[],
		const NNRT_CPU_LINALG_INT B_leading_dimension,
		const double beta,
		double* C_array[],
		const NNRT_CPU_LINALG_INT C_leading_dimension,
		const NNRT_CPU_LINALG_INT batch_size
) {
	//TODO: emulate parallelized batched BLAS routine via openMP for loop and regular gemm
#ifdef USE_BLAS
	open3d::utility::LogError("Not currently supported with usage of USE_BLAS (OpenBLAS + LAPACKE).");
#else
	cblas_dgemm_batch(layout, &transpose_A, &transpose_B, &m, &n, &k, &alpha, A_array, &A_leading_dimension, B_array,
	                  &B_leading_dimension, &beta, C_array, &C_leading_dimension, 1, &batch_size);
#endif
}


#ifdef BUILD_CUDA_MODULE

template<typename scalar_t>
inline cublasStatus_t gemm_batched_cuda(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m,
		int n,
		int k,
		const scalar_t* alpha,
		const scalar_t* A_array[], int A_leading_dimension,
		const scalar_t* B_array[], int B_leading_dimension,
		const scalar_t* beta,
		scalar_t* C_array[], int C_leading_dimension,
		int batchCount
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUBLAS_STATUS_NOT_SUPPORTED;
}

template<>
inline cublasStatus_t gemm_batched_cuda<float>(
		cublasHandle_t handle,
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const float* alpha,
		const float* Aarray[], int A_leading_dimension,
		const float* Barray[], int B_leading_dimension,
		const float* beta,
		float* Carray[], int C_leading_dimension,
		int batchCount
) {
	return cublasSgemmBatched(handle, transa, transb,
	                          m, n, k,
	                          alpha,
	                          Aarray, A_leading_dimension,
	                          Barray, B_leading_dimension,
	                          beta,
	                          Carray, C_leading_dimension, batchCount);
}

template<>
inline cublasStatus_t gemm_batched_cuda<double>(
		cublasHandle_t handle,
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const double* alpha,
		const double* Aarray[], int A_leading_dimension,
		const double* Barray[], int B_leading_dimension,
		const double* beta,
		double* Carray[], int C_leading_dimension,
		int batchCount
) {
	return cublasDgemmBatched(handle, transa, transb,
	                          m, n, k,
	                          alpha,
	                          Aarray, A_leading_dimension,
	                          Barray, B_leading_dimension,
	                          beta,
	                          Carray, C_leading_dimension, batchCount);
}

#endif

// endregion ==========================================================================================================

// region ============================= ?trsm =========================================================================
template<typename scalar_t>
inline void trsm(
		const CBLAS_LAYOUT layout, const CBLAS_SIDE A_equation_side,
		const CBLAS_UPLO upper_or_lower_triangle, const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_DIAG diagonal,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const scalar_t alpha,
		const scalar_t* A,
		//NOTE: number of columns, NOT number of rows, IFF layout == LAPACK_ROW_MAJOR
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		scalar_t* B,
		//NOTE: number of columns, NOT number of rows, IFF layout == LAPACK_ROW_MAJOR
		const NNRT_CPU_LINALG_INT B_leading_dimension
) {
	open3d::utility::LogError("Unsupported data type.");
}

template<>
inline void trsm<float>(
		const CBLAS_LAYOUT layout, const CBLAS_SIDE A_equation_side,
		const CBLAS_UPLO upper_or_lower_triangle, const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_DIAG diagonal,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const float alpha,
		const float* A,
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		float* B,
		const NNRT_CPU_LINALG_INT B_leading_dimension
) {
	cblas_strsm(layout, A_equation_side, upper_or_lower_triangle, transpose_A, diagonal,
	            m, n, alpha, A, A_leading_dimension, B, B_leading_dimension);
}

template<>
inline void trsm<double>(
		const CBLAS_LAYOUT layout, const CBLAS_SIDE A_equation_side,
		const CBLAS_UPLO upper_or_lower_triangle, const CBLAS_TRANSPOSE transpose_A,
		const CBLAS_DIAG diagonal,
		const NNRT_CPU_LINALG_INT m,
		const NNRT_CPU_LINALG_INT n,
		const double alpha,
		const double* A,
		const NNRT_CPU_LINALG_INT A_leading_dimension,
		double* B,
		const NNRT_CPU_LINALG_INT B_leading_dimension
) {
	cblas_dtrsm(layout, A_equation_side, upper_or_lower_triangle, transpose_A, diagonal,
	            m, n, alpha, A, A_leading_dimension, B, B_leading_dimension);
}
// endregion
} // nnrt::core