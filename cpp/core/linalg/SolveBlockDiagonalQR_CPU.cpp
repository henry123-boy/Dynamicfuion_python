//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/4/23.
//  Copyright (c) 2023 Gregory Kramida. All rights reserved.
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
// third-party includes
#include <open3d/core/linalg/LapackWrapper.h>
#include <open3d/utility/Parallel.h>
// local includes
#include "core/linalg/SolveBlockDiagonalQR.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"

namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<typename scalar_t, bool BAssumeFullRank>
inline void SolveBlockDiagonalQR_CPU_Generic(
        void *A_blocks_data,
        void *B_data,
        const int64_t A_and_B_block_row_count,
        const int64_t B_column_count,
        const int64_t block_count
) {
    auto *A_blocks_data_typed = static_cast<scalar_t *>(A_blocks_data);
    auto *B_data_typed = static_cast<scalar_t *>(B_data);
    const int64_t A_block_stride = A_and_B_block_row_count * A_and_B_block_row_count;
    const int64_t B_block_stride = A_and_B_block_row_count * B_column_count;
#pragma omp parallel for schedule(static) num_threads(utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(block_count, A_block_stride, A_and_B_block_row_count, B_block_stride, B_column_count) \
    shared(A_blocks_data_typed, B_data_typed)
    for (int64_t i_block = 0; i_block < block_count; i_block++) {
        auto *A_block_data = A_blocks_data_typed + A_block_stride * i_block;
        auto *B_block_data = B_data_typed + B_block_stride * i_block;
        if (BAssumeFullRank) {
            NNRT_LAPACK_CHECK(
                    open3d::core::gels_cpu<scalar_t>(LAPACK_COL_MAJOR, 'N', A_and_B_block_row_count,
                                                     A_and_B_block_row_count, B_column_count,
                                                     static_cast<scalar_t *>(A_block_data), A_and_B_block_row_count,
                                                     static_cast<scalar_t *>(B_block_data), A_and_B_block_row_count
                    ),
                    "gels failed in LeastSquaresCPU"
            );
        } else {
            long long rank;
            long long jbvt[A_and_B_block_row_count];
            memset(&jbvt, 0, sizeof(long long) * A_and_B_block_row_count);
            NNRT_LAPACK_CHECK(
                    nnrt::core::gelsy_cpu<scalar_t>(LAPACK_COL_MAJOR, A_and_B_block_row_count,
                                                    A_and_B_block_row_count, B_column_count,
                                                    static_cast<scalar_t *>(A_block_data),
                                                    A_and_B_block_row_count,
                                                    static_cast<scalar_t *>(B_block_data),
                                                    A_and_B_block_row_count,
                                                    jbvt, 1e-7, &rank
                    ),
                    "gelsy failed in LeastSquaresCPU"
            );
        }
    }
}

void SolveBlockDiagonalQR_CPU_FullRank(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
	    SolveBlockDiagonalQR_CPU_Generic<scalar_t, true>(A_blocks_data, B_data, A_and_B_block_row_count,
	                                                     B_column_count, block_count);
    });
}
void SolveBlockDiagonalQR_CPU_General(
        void *A_blocks_data,
        void *B_data,
        int64_t A_and_B_block_row_count,
        int64_t B_column_count,
        int64_t block_count,
        open3d::core::Dtype data_type,
        const open3d::core::Device &device
) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
	    SolveBlockDiagonalQR_CPU_Generic<scalar_t, false>(A_blocks_data, B_data, A_and_B_block_row_count,
	                                                      B_column_count, block_count);
    });
}
} // namespace nnrt::core::linalg::internal