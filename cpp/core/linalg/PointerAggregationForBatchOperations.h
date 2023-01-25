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

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_ABC(
		const scalar_t* A_array[], const scalar_t* B_array[], scalar_t* C_array[],
		const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
		const int64_t batch_size
) {
	auto A_data = static_cast<const scalar_t*>(A);
	auto B_data = static_cast<const scalar_t*>(B);
	auto C_data = static_cast<scalar_t*>(C);

	auto matrix_A_coefficient_count = m * k;
	auto matrix_B_coefficient_count = k * n;
	auto matrix_C_coefficient_count = m * n;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
	firstprivate(batch_size, matrix_A_coefficient_count, matrix_B_coefficient_count, matrix_C_coefficient_count) \
	shared(A_data, B_data, C_data, A_array, B_array, C_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * matrix_A_coefficient_count;
		B_array[i_matrix] = B_data + i_matrix * matrix_B_coefficient_count;
		C_array[i_matrix] = C_data + i_matrix * matrix_C_coefficient_count;
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

	auto matrix_A_coefficient_count = A_and_B_row_count * A_and_B_row_count;
	auto matrix_B_coefficient_count = A_and_B_row_count * B_column_count;

#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
	firstprivate(batch_size, matrix_A_coefficient_count, matrix_B_coefficient_count) \
	shared(A_data, B_data, A_array, B_array)
	for (int i_matrix = 0; i_matrix < batch_size; i_matrix++) {
		A_array[i_matrix] = A_data + i_matrix * matrix_A_coefficient_count;
		B_array[i_matrix] = B_data + i_matrix * matrix_B_coefficient_count;
	}
}