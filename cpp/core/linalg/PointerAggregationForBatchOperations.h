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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/utility/Parallel.h>
#include <cstdint>

// local includes
namespace nnrt::core::linalg::internal {
//TODO: come up with CUDA versions of these routines (for large batch sizes)
template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_ABC(
		const scalar_t* A_array[], const scalar_t* B_array[], scalar_t* C_array[],
		const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
		const int64_t batch_size
) {
	auto A_data = static_cast<const scalar_t*>(A);
	auto B_data = static_cast<const scalar_t*>(B);
	auto C_data = static_cast<scalar_t*>(C);

	auto A_block_stride = m * k;
	auto B_block_stride = k * n;
	auto C_block_stride = m * n;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(batch_size, A_block_stride, B_block_stride, C_block_stride) \
    shared(A_data, B_data, C_data, A_array, B_array, C_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * A_block_stride;
		B_array[i_matrix] = B_data + i_matrix * B_block_stride;
		C_array[i_matrix] = C_data + i_matrix * C_block_stride;
	}
}

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_AB(
		scalar_t* A_array[], scalar_t* B_array[],
		void* A, void* B,
		const int64_t A_and_B_row_count,
		const int64_t B_column_count,
		const int64_t batch_size
) {
	auto A_data = static_cast<scalar_t*>(A);
	auto B_data = static_cast<scalar_t*>(B);

	auto A_block_stride = A_and_B_row_count * A_and_B_row_count;
	auto B_block_stride = A_and_B_row_count * B_column_count;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(batch_size, A_block_stride, B_block_stride) \
    shared(A_data, B_data, A_array, B_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * A_block_stride;
		B_array[i_matrix] = B_data + i_matrix * B_block_stride;
	}
}

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_constA_B(
		const scalar_t* A_array[], scalar_t* B_array[],
		const void* A, void* B,
		const int64_t A_and_B_row_count,
		const int64_t B_column_count,
		const int64_t batch_size
) {
	auto A_data = static_cast<const scalar_t*>(A);
	auto B_data = static_cast<scalar_t*>(B);

	auto A_block_stride = A_and_B_row_count * A_and_B_row_count;
	auto B_block_stride = A_and_B_row_count * B_column_count;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(batch_size, A_block_stride, B_block_stride) \
    shared(A_data, B_data, A_array, B_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * A_block_stride;
		B_array[i_matrix] = B_data + i_matrix * B_block_stride;
	}
}


template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices(
		scalar_t* A_array[],
		void* A,
		const int64_t A_row_count,
		const int64_t A_column_count,
		const int64_t batch_size
) {
	auto A_data = static_cast<scalar_t*>(A);

	auto A_block_stride = A_row_count * A_column_count;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(batch_size, A_block_stride) \
    shared(A_data, A_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * A_block_stride;
	}
}
} // namespace nnrt::core::linalg::internal