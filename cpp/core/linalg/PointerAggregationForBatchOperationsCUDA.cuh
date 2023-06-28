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
#include <open3d/core/ParallelFor.h>
// local
#include "core/platform_independence/Qualifiers.h"

namespace utility = open3d::utility;
namespace o3c = open3d::core;

// local includes
namespace nnrt::core::linalg::internal {
//TODO: come up with CUDA versions of these routines (for large batch sizes)
template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_ABC_CUDA(
		const scalar_t* A_array[], const scalar_t* B_array[], scalar_t* C_array[],
		const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
		const int64_t batch_size
) {
	utility::LogError("Not implemented");
}

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_AB_CUDA(
		scalar_t* A_array[], scalar_t* B_array[],
		void* A, void* B,
		const int64_t A_and_B_row_count,
		const int64_t B_column_count,
		const int64_t batch_size
) {
	utility::LogError("Not implemented");
}

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_constA_B_CUDA(
		const scalar_t* A_array[], scalar_t* B_array[],
		const void* A, void* B,
		const int64_t A_and_B_row_count,
		const int64_t B_column_count,
		const int64_t batch_size
) {
	utility::LogError("Not implemented");
}


template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatrices_CUDA(
		scalar_t* A_array[],
		void* A,
		const int64_t A_row_count,
		const int64_t A_column_count,
		const int64_t batch_size,
		const o3c::Device& device
) {
	auto A_data = static_cast<scalar_t*>(A);

	auto A_block_stride = A_row_count * A_column_count;
	o3c::ParallelFor(
			device,
			batch_size,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_batch) {
				A_array[i_batch] = A_data + i_batch * A_block_stride;
			}
	);
}

template<typename scalar_t>
inline void GetMatrixPointersFromContiguousArrayOfMatricesPadded_CUDA(
		scalar_t* A_array[],
		void* A,
		void* A_padding,
		const int64_t A_row_count,
		const int64_t A_column_count,
		const int64_t batch_size,
		const int64_t padding_size,
		const o3c::Device& device
) {
	auto A_data = static_cast<scalar_t*>(A);
	auto A_padding_data = static_cast<scalar_t*>(A_padding);

	auto A_block_stride = A_row_count * A_column_count;
	o3c::ParallelFor(
			device,
			batch_size + padding_size,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_batch) {
				if (i_batch < batch_size) {
					A_array[i_batch] = A_data + i_batch * A_block_stride;
				} else {
					int64_t i_batch_padding = i_batch - batch_size;
					A_array[i_batch] = A_padding_data + i_batch_padding * A_block_stride;
				}
			}
	);
}

} // namespace nnrt::core::linalg::internal