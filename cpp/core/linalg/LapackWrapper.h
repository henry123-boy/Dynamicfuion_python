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

// contains some blas operations currently missing from Open3D LapackWrapper.h,
// i.e. ?potrf (Cholesky factorization of a symmetric (Hermitian) positive-definite matrix)

namespace nnrt::core {

template<typename scalar_t>
inline NNRT_CPU_LINALG_INT
potrf_cpu(int layout, char upper_or_lower, NNRT_CPU_LINALG_INT A_leading_dimension, scalar_t* A_data, NNRT_CPU_LINALG_INT A_other_dimension) {
	open3d::utility::LogError("Unsupported data type.");
	return -1;
}

template<>
inline NNRT_CPU_LINALG_INT potrf_cpu<float>(
		int layout,
		char upper_or_lower,
		NNRT_CPU_LINALG_INT A_leading_dimension,
		float* A_data,
		NNRT_CPU_LINALG_INT A_other_dimension
) {
	return LAPACKE_spotrf(layout, upper_or_lower, A_other_dimension, A_data, A_leading_dimension);
}

template<>
inline NNRT_CPU_LINALG_INT potrf_cpu<double>(
		int layout,
		char upper_or_lower,
		NNRT_CPU_LINALG_INT A_leading_dimension,
		double* A_data,
		NNRT_CPU_LINALG_INT A_other_dimension
) {
	return LAPACKE_dpotrf(layout, upper_or_lower, A_other_dimension, A_data, A_leading_dimension);
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
	return cusolverDnSpotrfBatched(handle, upper_or_lower_triangle, n, A_array, A_leading_dimension, out_factorization_result_array, batch_size);
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
	return cusolverDnDpotrfBatched(handle, upper_or_lower_triangle, n, A_array, A_leading_dimension, out_factorization_result_array, batch_size);
}
#endif


} // nnrt::core