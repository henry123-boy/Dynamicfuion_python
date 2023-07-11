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
		cublasHandle_t
handle,
cublasOperation_t transpose_A,
		cublasOperation_t
transpose_B,
int m,
int n,
int k,
const scalar_t* alpha,
const scalar_t* A_array[],
int A_leading_dimension,
const scalar_t* B_array[],
int B_leading_dimension,
const scalar_t* beta,
		scalar_t
* C_array[],
int C_leading_dimension,
int batch_count
) {
open3d::utility::LogError("Unsupported data type.");
return
CUBLAS_STATUS_NOT_SUPPORTED;
}

template<>
inline cublasStatus_t gemm_batched_cuda<float>(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m,
		int n,
		int k,
		const float* alpha,
		const float* A_array[], int A_leading_dimension,
		const float* B_array[], int B_leading_dimension,
		const float* beta,
		float* Carray[], int C_leading_dimension,
		int batch_count
) {
	return cublasSgemmBatched(handle, transpose_A, transpose_B,
	                          m, n, k,
	                          alpha,
	                          A_array, A_leading_dimension,
	                          B_array, B_leading_dimension,
	                          beta,
	                          Carray, C_leading_dimension, batch_count);
}

template<>
inline cublasStatus_t gemm_batched_cuda<double>(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m,
		int n,
		int k,
		const double* alpha,
		const double* A_array[], int A_leading_dimension,
		const double* B_array[], int B_leading_dimension,
		const double* beta,
		double* Carray[], int C_leading_dimension,
		int batch_count
) {
	return cublasDgemmBatched(handle, transpose_A, transpose_B,
	                          m, n, k,
	                          alpha,
	                          A_array, A_leading_dimension,
	                          B_array, B_leading_dimension,
	                          beta,
	                          Carray, C_leading_dimension, batch_count);
}

#endif

// endregion ==========================================================================================================

// region ============================= ?trsm =========================================================================
template<typename scalar_t>
inline void trsm_cpu(
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
inline void trsm_cpu<float>(
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
#ifdef USE_BLAS
	open3d::utility::LogError("Not currently supported with usage of USE_BLAS (OpenBLAS + LAPACKE).");
#else
	cblas_strsm(layout, A_equation_side, upper_or_lower_triangle, transpose_A, diagonal,
	            m, n, alpha, A, A_leading_dimension, B, B_leading_dimension);
#endif
}

template<>
inline void trsm_cpu<double>(
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
#ifdef USE_BLAS
	open3d::utility::LogError("Not currently supported with usage of USE_BLAS (OpenBLAS + LAPACKE).");
#else
	cblas_dtrsm(layout, A_equation_side, upper_or_lower_triangle, transpose_A, diagonal,
	            m, n, alpha, A, A_leading_dimension, B, B_leading_dimension);
#endif
}

#ifdef BUILD_CUDA_MODULE

template<typename scalar_t>
inline cublasStatus_t trsm_cuda(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int m,
		int n,
		const scalar_t* alpha,
		const scalar_t* A,
		int lda, // column leading dimension
		scalar_t* B,
		int ldb
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUBLAS_STATUS_NOT_SUPPORTED;
}

template<>
inline cublasStatus_t trsm_cuda<float>(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int m,
		int n,
		const float* alpha,
		const float* A,
		int lda, // column leading dimension
		float* B,
		int ldb
) {
	return cublasStrsm(
			handle,
			side, upper_or_lower_triangle,
			transpose_A, A_diagonal_type,
			m, n,
			alpha,
			A, lda,
			B, ldb
	);
}

template<>
inline cublasStatus_t trsm_cuda<double>(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int m,
		int n,
		const double* alpha,
		const double* A,
		int lda, // column leading dimension
		double* B,
		int ldb
) {
	return cublasDtrsm(
			handle,
			side, upper_or_lower_triangle,
			transpose_A, A_diagonal_type,
			m, n,
			alpha,
			A, lda,
			B, ldb
	);
}

template<typename scalar_t>
inline cublasStatus_t trsm_batched_cuda(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int B_row_count,
		int B_column_count,
		const scalar_t* alpha,
		const scalar_t* const A_array[],
		int A_leading_dimension, // column count
		scalar_t* const B_array[],
		int B_leading_dimension,
		int batch_count
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUBLAS_STATUS_NOT_SUPPORTED;
}

template<>
inline cublasStatus_t trsm_batched_cuda<float>(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int B_row_count,
		int B_column_count,
		const float* alpha,
		const float* const A_array[],
		int A_leading_dimension, // column count
		float* const B_array[],
		int B_leading_dimension,
		int batch_count
) {
	return cublasStrsmBatched(handle, side, upper_or_lower_triangle,
	                          transpose_A, A_diagonal_type,
	                          B_row_count, B_column_count,
	                          alpha, A_array, A_leading_dimension,
	                          B_array, B_leading_dimension,
	                          batch_count);
}

template<>
inline cublasStatus_t trsm_batched_cuda<double>(
		cublasHandle_t handle,
		cublasSideMode_t side,
		cublasFillMode_t upper_or_lower_triangle,
		cublasOperation_t transpose_A,
		cublasDiagType_t A_diagonal_type,
		int B_row_count,
		int B_column_count,
		const double* alpha,
		const double* const A_array[],
		int A_leading_dimension, // column count
		double* const B_array[],
		int B_leading_dimension,
		int batch_count
) {
	return cublasDtrsmBatched(handle, side, upper_or_lower_triangle,
	                          transpose_A, A_diagonal_type,
	                          B_row_count, B_column_count,
	                          alpha, A_array, A_leading_dimension,
	                          B_array, B_leading_dimension,
	                          batch_count);
}


#endif

// endregion
// region ============================= ?trmm: Matrix product with one of the arguments being a triangular matrix ====================================
// region ----- CPU -------
template<typename scalar_t>
inline void trmm_cpu(
		const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
		const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
		const NNRT_CPU_LINALG_INT M,
		const NNRT_CPU_LINALG_INT N,
		const scalar_t alpha,
		const scalar_t* A, const NNRT_CPU_LINALG_INT lda,
		scalar_t* B, const NNRT_CPU_LINALG_INT ldb
) {
	open3d::utility::LogError("Unsupported data type.");
}

template<>
inline void trmm_cpu<float>(
		const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
		const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
		const NNRT_CPU_LINALG_INT M, const NNRT_CPU_LINALG_INT N,
		const float alpha,
		const float* A, const NNRT_CPU_LINALG_INT lda,
		float* B, const NNRT_CPU_LINALG_INT ldb
) {
	cblas_strmm(
			Layout, Side, Uplo,
			TransA, Diag,
			M, N,
			alpha,
			A, lda,
			B, ldb
	);
}

template<>
inline void trmm_cpu<double>(
		const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
		const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
		const NNRT_CPU_LINALG_INT M, const NNRT_CPU_LINALG_INT N,
		const double alpha,
		const double* A, const NNRT_CPU_LINALG_INT lda,
		double* B, const NNRT_CPU_LINALG_INT ldb
) {
	cblas_dtrmm(
			Layout, Side, Uplo,
			TransA, Diag,
			M, N,
			alpha,
			A, lda,
			B, ldb
	);
}


// endregion
// region ----- CUDA ------
#ifdef BUILD_CUDA_MODULE

template<typename scalar_t>
inline void trmm_batched_cuda_inplace(
		magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
		magma_int_t m, magma_int_t n,
		scalar_t alpha,
		scalar_t** dA_array, magma_int_t ldda,
		scalar_t** dB_array, magma_int_t lddb,
		magma_int_t batchCount, magma_queue_t queue
) {
	open3d::utility::LogError("Unsupported data type.");
}

template<>
inline void trmm_batched_cuda_inplace<float>(
		magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
		magma_int_t m, magma_int_t n,
		float alpha,
		float** dA_array, magma_int_t ldda,
		float** dB_array, magma_int_t lddb,
		magma_int_t batchCount, magma_queue_t queue
) {
	magmablas_strmm_batched(
			side, uplo, transA, diag,
			m, n,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			batchCount, queue
	);
}

template<>
inline void trmm_batched_cuda_inplace<double>(
		magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
		magma_int_t m, magma_int_t n,
		double alpha,
		double** dA_array, magma_int_t ldda,
		double** dB_array, magma_int_t lddb,
		magma_int_t batchCount, magma_queue_t queue
) {
	magmablas_dtrmm_batched(
			side, uplo, transA, diag,
			m, n,
			alpha,
			dA_array, ldda,
			dB_array, lddb,
			batchCount, queue
	);
}


// endregion
#endif
// endregion
// region =============================================== ?geam : matrix-matrix addition/transposition ===============================================
// region ==== CUDA ====
#ifdef BUILD_CUDA_MODULE

template<typename scalar_t>
inline cublasStatus_t geam_cuda(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m, int n,
		const scalar_t* alpha,
		const scalar_t* A, int lda,
		const scalar_t* beta,
		const scalar_t* B, int ldb,
		scalar_t* C, int ldc
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUBLAS_STATUS_NOT_SUPPORTED;
}

template<>
inline cublasStatus_t geam_cuda<float>(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m, int n,
		const float* alpha,
		const float* A, int lda,
		const float* beta,
		const float* B, int ldb,
		float* C, int ldc
) {
	return cublasSgeam(handle, transpose_A, transpose_B, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template<>
inline cublasStatus_t geam_cuda<double>(
		cublasHandle_t handle,
		cublasOperation_t transpose_A,
		cublasOperation_t transpose_B,
		int m, int n,
		const double* alpha,
		const double* A, int lda,
		const double* beta,
		const double* B, int ldb,
		double* C, int ldc
) {
	return cublasDgeam(handle, transpose_A, transpose_B, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

#endif
// endregion
// endregion
// region ========================================================== imatcopy (compute in-place scaled matrix copy or transpoistion) =================
template<typename scalar_t>
inline void imatcopy_batched_cpu(
		char layout, const char* trans_array,
		const size_t* rows_array, const size_t* cols_array,
		const float* alpha_array, 
		float** ab_array, const size_t* lda_array, 
		const size_t* ldb_array,
		size_t group_count, const size_t* group_size
) {
	open3d::utility::LogError("Unsupported data type.");
}

template<>
inline void imatcopy_batched_cpu<float>(
		char layout, const char* trans_array,
		const size_t* rows_array, const size_t* cols_array,
		const float* alpha_array,
		float** ab_array, const size_t* lda_array,
		const size_t* ldb_array,
		size_t group_count, const size_t* group_size
) {
	//TODO
}

// ===================================================================================================================================================

} // nnrt::core