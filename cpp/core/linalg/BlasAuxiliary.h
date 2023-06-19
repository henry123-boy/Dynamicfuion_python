//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/19/23.
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
#include <open3d/core/MemoryManager.h>

// local includes
#include "core/linalg/BlasWrapper.h"

/**
 * \brief Header for auxiliary Blas routines, e.g. batched versions of things for which batched versions are not readily available, etc.
 */

namespace nnrt::core {

#ifndef __CUDACC__

template<typename scalar_t>
inline void trmm_batched_cpu_inplace(
		const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
		const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
		NNRT_CPU_LINALG_INT M, NNRT_CPU_LINALG_INT N,
		const scalar_t alpha,
		const scalar_t** dA_array, NNRT_CPU_LINALG_INT ldda,
		scalar_t** dB_array, NNRT_CPU_LINALG_INT lddb,
		NNRT_CPU_LINALG_INT batchCount
) {
#pragma omp parallel for default(none) shared(dA_array, dB_array, batchCount) firstprivate(Layout, Side, Uplo, TransA, Diag, M, N, ldda, lddb, alpha)
	for (int i_batch = 0; i_batch < batchCount; i_batch++) {
		trmm_cpu<scalar_t>(
				Layout, Side, Uplo,
				TransA, Diag,
				M, N,
				alpha,
				dA_array[i_batch], ldda,
				dB_array[i_batch], lddb
		);
	}
};

template<typename scalar_t>
inline void trmm_batched_cpu(
		const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo,
		const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
		NNRT_CPU_LINALG_INT M, NNRT_CPU_LINALG_INT N,
		const scalar_t alpha,
		const scalar_t** dA_array, NNRT_CPU_LINALG_INT ldda,
		const scalar_t** dB_array, NNRT_CPU_LINALG_INT lddb,
		scalar_t** dC_array, NNRT_CPU_LINALG_INT lddc,
		NNRT_CPU_LINALG_INT batchCount
) {
	if (lddc < M) {
		open3d::utility::LogError("ldc ({}) must be at least max(1, M) ({}).", lddc, M);
	}
	if (lddb == lddc || Layout == CBLAS_LAYOUT::CblasRowMajor) {
		NNRT_CPU_LINALG_INT array_bytesize = sizeof(scalar_t) * lddc * N;
#pragma omp parallel for default(none) shared(dB_array, dC_array, batchCount) firstprivate(array_bytesize)
		for (int i_batch = 0; i_batch < batchCount; i_batch++) {
			memcpy(dC_array[i_batch], dB_array[i_batch], array_bytesize);
		}
	} else {
		NNRT_CPU_LINALG_INT column_bytesize = M * sizeof(scalar_t);
#pragma omp parallel for default(none) shared(dB_array, dC_array, batchCount) firstprivate(column_bytesize, N, lddb, lddc)
		for (NNRT_CPU_LINALG_INT i_column = 0; i_column < batchCount * N; i_column++) {
			NNRT_CPU_LINALG_INT i_batch = i_column / N;
			NNRT_CPU_LINALG_INT i_column_in_batch = i_column % N;
			memcpy(dC_array[i_batch] + i_column_in_batch * lddc, dB_array[i_batch] + i_column_in_batch * lddb, column_bytesize);
		}
	}
	trmm_batched_cpu_inplace(Layout, Side, Uplo,
	                         TransA, Diag,
	                         M, N, alpha,
	                         dA_array, ldda,
	                         dC_array, lddc,
	                         batchCount);
};
#endif

} // namespace nnrt::core