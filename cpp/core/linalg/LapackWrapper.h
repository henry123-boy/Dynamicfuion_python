//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/23/23.
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
#include <open3d/utility/Logging.h>

// local includes
#include "core/linalg/LinalgHeadersCUDA.h"
#include "core/linalg/LinalgHeadersCPU.h"
#include <open3d/core/CUDAUtils.h>

// contains some blas operations currently missing from Open3D LapackWrapper.h,
// i.e. ?potrf (Cholesky factorization of a symmetric (Hermitian) positive-definite matrix)

namespace nnrt::core {
// region =============================== POTRF: Cholesky Matrix Factorization =======================================================================
template<typename scalar_t>
inline NNRT_CPU_LINALG_INT
potrf_cpu(
		int layout,
		char upper_or_lower,
		NNRT_CPU_LINALG_INT A_leading_dimension,
		scalar_t* A_data,
		NNRT_CPU_LINALG_INT A_row_and_column_count
) {
	open3d::utility::LogError("Unsupported data type.");
	return -1;
}

template<>
inline NNRT_CPU_LINALG_INT potrf_cpu<float>(
		int layout,
		char upper_or_lower,
		NNRT_CPU_LINALG_INT A_leading_dimension,
		float* A_data,
		NNRT_CPU_LINALG_INT A_row_and_column_count
) {
	return LAPACKE_spotrf(layout, upper_or_lower, A_row_and_column_count, A_data, A_leading_dimension);
}

template<>
inline NNRT_CPU_LINALG_INT potrf_cpu<double>(
		int layout,
		char upper_or_lower,
		NNRT_CPU_LINALG_INT A_leading_dimension,
		double* A_data,
		NNRT_CPU_LINALG_INT A_row_and_column_count
) {
	return LAPACKE_dpotrf(layout, upper_or_lower, A_row_and_column_count, A_data, A_leading_dimension);
}


#ifdef BUILD_CUDA_MODULE

// See https://docs.nvidia.com/cuda/cusolver/#cusolverdn-t-potrfbatched
template<typename scalar_t>
inline cusolverStatus_t potrf_batched_cuda(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int n,
		scalar_t* A_array[],
		int A_leading_dimension,
		int* out_factorization_result_array,
		int batch_size
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUSOLVER_STATUS_NOT_SUPPORTED;
}

template<>
inline cusolverStatus_t potrf_batched_cuda<float>(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int n,
		float* A_array[],
		int A_leading_dimension,
		int* out_factorization_result_array,
		int batch_size
) {
	return cusolverDnSpotrfBatched(handle, upper_or_lower_triangle, n, A_array, A_leading_dimension,
	                               out_factorization_result_array, batch_size);
}

template<>
inline cusolverStatus_t potrf_batched_cuda<double>(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int n,
		double* A_array[],
		int A_leading_dimension,
		int* out_factorization_result_array,
		int batch_size
) {
	return cusolverDnDpotrfBatched(handle, upper_or_lower_triangle, n, A_array, A_leading_dimension,
	                               out_factorization_result_array, batch_size);
}

template<typename scalar_t>
inline cusolverStatus_t potrf_cuda(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int A_leading_dimension,
		scalar_t* A_data,
		int A_row_and_column_count
) {
	open3d::utility::LogError("Unsupported data type.");
	return CUSOLVER_STATUS_NOT_SUPPORTED;
}

namespace internal {
template<typename scalar_t, typename TDetermineBufferSize, typename TPotrf>
inline cusolverStatus_t potrf_cuda_generic(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int A_leading_dimension,
		scalar_t* A_data,
		int A_row_and_column_count,
		TDetermineBufferSize&& determine_buffer_size,
		TPotrf&& potrf
) {
	int l_work = 0;
	cusolverStatus_t buffer_size_error = determine_buffer_size(handle, upper_or_lower_triangle, A_row_and_column_count, A_data,
	                                                           A_leading_dimension, &l_work);
	if (CUSOLVER_STATUS_SUCCESS != buffer_size_error) {
		return buffer_size_error;
	}
	scalar_t* workspace;
	int* info_device;
	OPEN3D_CUDA_CHECK(cudaMalloc(&workspace, sizeof(scalar_t) * l_work));
	OPEN3D_CUDA_CHECK(cudaMalloc(&info_device, sizeof(int)));
	OPEN3D_CUDA_CHECK(cudaMemset(info_device, 0, sizeof(int)));
	cusolverStatus_t potrf_status = potrf(
			handle, CUBLAS_FILL_MODE_UPPER, A_row_and_column_count,
			A_data, A_leading_dimension,
			workspace, l_work, info_device
	);
	OPEN3D_CUDA_CHECK(cudaFree(workspace));
	OPEN3D_CUDA_CHECK(cudaFree(info_device));
	return potrf_status;
}
} // namespace internal


template<>
inline cusolverStatus_t potrf_cuda<float>(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int A_leading_dimension,
		float* A_data,
		int A_row_and_column_count
) {
	return internal::potrf_cuda_generic(handle, upper_or_lower_triangle, A_leading_dimension, A_data, A_row_and_column_count,
	                                    cusolverDnSpotrf_bufferSize, cusolverDnSpotrf);
}


template<>
inline cusolverStatus_t potrf_cuda<double>(
		cusolverDnHandle_t handle,
		cublasFillMode_t upper_or_lower_triangle,
		int A_leading_dimension,
		double* A_data,
		int A_row_and_column_count
) {
	return internal::potrf_cuda_generic(handle, upper_or_lower_triangle, A_leading_dimension, A_data, A_row_and_column_count,
	                                    cusolverDnDpotrf_bufferSize, cusolverDnDpotrf);
}

#endif

// endregion =========================================================================================================================================
// region ================================ GELSY: linear equation solver using rank-revealing QR factorization =======================================
template<typename scalar_t>
inline NNRT_CPU_LINALG_INT gelsy_cpu(
		int matrix_layout,
		NNRT_CPU_LINALG_INT m,
		NNRT_CPU_LINALG_INT n,
		NNRT_CPU_LINALG_INT nrhs,
		scalar_t* A_data,
		NNRT_CPU_LINALG_INT lda,
		scalar_t* B_data,
		NNRT_CPU_LINALG_INT ldb,
		NNRT_CPU_LINALG_INT* jbvt,
		scalar_t rcond,
		NNRT_CPU_LINALG_INT* rank
) {
	open3d::utility::LogError("Unsupported data type.");
	return -1;
}

template<>
inline NNRT_CPU_LINALG_INT gelsy_cpu<float>(
		int matrix_layout,
		NNRT_CPU_LINALG_INT m,
		NNRT_CPU_LINALG_INT n,
		NNRT_CPU_LINALG_INT nrhs,
		float* A_data,
		NNRT_CPU_LINALG_INT lda,
		float* B_data,
		NNRT_CPU_LINALG_INT ldb,
		NNRT_CPU_LINALG_INT* jbvt,
		float rcond,
		NNRT_CPU_LINALG_INT* rank
) {
	return LAPACKE_sgelsy(matrix_layout, m, n, nrhs, A_data, lda, B_data, ldb, jbvt, rcond, rank);
}

template<>
inline NNRT_CPU_LINALG_INT gelsy_cpu<double>(
		int matrix_layout,
		NNRT_CPU_LINALG_INT m,
		NNRT_CPU_LINALG_INT n,
		NNRT_CPU_LINALG_INT nrhs,
		double* A_data,
		NNRT_CPU_LINALG_INT lda,
		double* B_data,
		NNRT_CPU_LINALG_INT ldb,
		NNRT_CPU_LINALG_INT* jbvt,
		double rcond,
		NNRT_CPU_LINALG_INT* rank
) {
	return LAPACKE_dgelsy(matrix_layout, m, n, nrhs, A_data, lda, B_data, ldb, jbvt, rcond, rank);
}
// endregion
// region ================================ GEQP3: rank-revealing QR factorization with column-pivoting ===============================================
// TODO: port from MAGMA or remove region
// endregion =========================================================================================================================================

// region ==================================== ?TRTRI: inversion of triangluar matrices (non-batched for CPU, batched for CUDA) ======================

// region --- CPU ---
template<typename scalar_t>
inline NNRT_CPU_LINALG_INT
trtri_cpu(
		int matrix_layout,
		char uplo,
		char diag,
		NNRT_CPU_LINALG_INT n,
		scalar_t* a,
		NNRT_CPU_LINALG_INT lda
) {
	open3d::utility::LogError("Unsupported data type.");
	return -1;
}

template<>
inline NNRT_CPU_LINALG_INT
trtri_cpu<float>(
		int matrix_layout,
		char uplo,
		char diag,
		NNRT_CPU_LINALG_INT n,
		float* a,
		NNRT_CPU_LINALG_INT lda
) {
	return LAPACKE_strtri(matrix_layout, uplo, diag, n, a, lda);
}

template<>
inline NNRT_CPU_LINALG_INT
trtri_cpu<double>(
		int matrix_layout,
		char uplo,
		char diag,
		NNRT_CPU_LINALG_INT n,
		double* a,
		NNRT_CPU_LINALG_INT lda
) {
	return LAPACKE_dtrtri(matrix_layout, uplo, diag, n, a, lda);
}
// endregion
// region --- CUDA ---
#ifdef BUILD_CUDA_MODULE

//deprecated("See by trtri_batched_cuda in LinalgKernels.cuh, this is probably NOT what you want")]]
template<typename scalar_t>
inline void trtri_diag_batched_cuda(
		magma_uplo_t uplo,
		magma_diag_t diag,
		magma_int_t n,
		scalar_t const* const* dA_array,
		magma_int_t ldda,
		scalar_t** dinvA_array,
		magma_int_t resetozero,
		magma_int_t batchCount,
		magma_queue_t queue
) {
	open3d::utility::LogError("Unsupported data type.");
}


template<>
inline void trtri_diag_batched_cuda<float>(
		magma_uplo_t uplo,
		magma_diag_t diag,
		magma_int_t n,
		float const* const* dA_array,
		magma_int_t ldda,
		float** dinvA_array,
		magma_int_t resetozero,
		magma_int_t batchCount,
		magma_queue_t queue
) {
	magmablas_strtri_diag_batched(uplo, diag, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue);
}

template<>
inline void trtri_diag_batched_cuda<double>(
		magma_uplo_t uplo,
		magma_diag_t diag,
		magma_int_t n,
		double const* const* dA_array,
		magma_int_t ldda,
		double** dinvA_array,
		magma_int_t resetozero,
		magma_int_t batchCount,
		magma_queue_t queue
) {
	magmablas_dtrtri_diag_batched(uplo, diag, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue);
}


template<typename scalar_t>
inline void trtri_cuda(
		cusolverDnHandle_t handle,
		cusolver_int_t n,
		scalar_t* d_A,
		cusolver_int_t lda,
		cublasFillMode_t uplo,
		cublasDiagType_t diag
) {
	void* d_work = nullptr;
	size_t d_lwork = 0;
	void* h_work = nullptr;
	size_t h_lwork = 0;

	int* info_device;
	OPEN3D_CUDA_CHECK(cudaMalloc(&info_device, sizeof(int)));

	try {
		NNRT_CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(
				handle,
				uplo,
				diag,
				n,
				cusolver_traits<scalar_t>::cuda_data_type,
				(void*) d_A,
				lda,
				&d_lwork,
				&h_lwork), "cusolverDnXtrtri_bufferSize failed.");
		OPEN3D_CUDA_CHECK(cudaMalloc((void**) &d_work, d_lwork));
		if (h_lwork) {
			h_work = malloc(h_lwork);
			if (h_work == nullptr) {
				throw std::bad_alloc();
			}
		}
		NNRT_CUSOLVER_CHECK(cusolverDnXtrtri(
				handle,
				uplo,
				diag,
				n,
				cusolver_traits<scalar_t>::cuda_data_type,
				d_A,
				lda,
				d_work,
				d_lwork,
				h_work,
				h_lwork,
				info_device), "cusolverDnXtrtri failed.");
	}
	catch (const std::exception& e) {
		fprintf(stderr, "error: %s\n", e.what());
	}

	OPEN3D_CUDA_CHECK(cudaFree(info_device));
	if (d_work) OPEN3D_CUDA_CHECK(cudaFree(d_work));
	if (h_lwork && h_work) free(h_work);
}

//endregion


#endif


// endregion
} // nnrt::core