//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/2/23.
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
// stdlib includes

// third-party includes
#include <open3d/core/Dtype.h>
#include <open3d/core/Device.h>
#include <open3d/utility/Parallel.h>

// local includes
#include "core/linalg/FactorizeBlocksCholesky.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"

namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<typename scalar_t>
void FactorizeBlocksCholeskyCPU_Generic(
		void* block_data,
		int64_t block_size,
		int64_t block_count
) {
	auto* block_data_typed = static_cast<scalar_t*>(block_data);
	const int64_t block_stride = block_size * block_size;

#pragma omp parallel for schedule(static) num_threads(utility::EstimateMaxThreads()) \
    default(none) \
	firstprivate(block_count, block_stride, block_size) \
	shared(block_data_typed)
	for (int64_t i_block = 0; i_block < block_count; i_block++) {
		auto* A_block_data = block_data_typed + block_stride * i_block;
		// use Cholesky factorization to compute lower-triangular L, where L(L^T) = A
		NNRT_LAPACK_CHECK(
				potrf_cpu<scalar_t>(
						LAPACK_COL_MAJOR, 'U', block_size, A_block_data, block_size),
				"potrf failed in SolveCholeskyBlockDiagonalCPU"
		);
	}
}

void FactorizeBlocksCholeskyCPU(
		void* block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		FactorizeBlocksCholeskyCPU_Generic<scalar_t>(
				block_data,
				block_size,
				block_count
		);
	});
}

} // namespace nnrt::core::linalg::internal
