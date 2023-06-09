//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/5/23.
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
#include <open3d/core/CUDAUtils.h>

// local includes
#include "core/linalg/InvertBlocks.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/PointerAggregationForBatchOperationsCUDA.cuh"
#include "core/CUDAUtils.h"


namespace utility = open3d::utility;
namespace o3c = open3d::core;

namespace nnrt::core::linalg::internal {


namespace {

template<typename scalar_t, typename TInvertNonDiagonal, typename TCopyTriangular>
//square matrices are assumed
__device__
inline void tritri_batched_generic(
		scalar_t** matrices, int matrix_size, int matrix_pair_count_per_block,
		int jobs_per_block, int global_thread_cutoff, TInvertNonDiagonal&& invert_non_diagonal, TCopyTriangular&& copy_triangular
) {
	int i_thread_in_block = static_cast<int>(threadIdx.x);
	int i_block = static_cast<int>(blockIdx.x);
	int block_size = static_cast<int>(blockDim.x);
	int i_thread = i_thread_in_block + i_block * block_size;
	if (i_thread >= global_thread_cutoff || i_thread_in_block >= jobs_per_block) {
		return;
	}
	extern __shared__ char shared_block_data_raw[];
	auto shared_block_data = reinterpret_cast<scalar_t*>(shared_block_data_raw);

	const int i_matrix_pair_in_block = i_thread_in_block / matrix_size;
	const int i_matrix_pair = i_block * matrix_pair_count_per_block + i_matrix_pair_in_block;

	const int i_matrix_a = i_matrix_pair * 2;
	const int i_column_in_a = i_thread % matrix_size;
	scalar_t* matrix_a = matrices[i_matrix_a];

	const int i_matrix_b = i_matrix_a + 1;
	const int i_column_in_b = matrix_size - i_column_in_a - 1;
	scalar_t* matrix_b = matrices[i_matrix_b];

	const int matrix_element_count = matrix_size * matrix_size;
	scalar_t* matrix_a_shared = shared_block_data + (i_matrix_pair_in_block * 4 * matrix_element_count);
	scalar_t* matrix_b_shared = matrix_a_shared + matrix_element_count;
	scalar_t* matrix_a_inverted_shared = matrix_b_shared + matrix_element_count;
	scalar_t* matrix_b_inverted_shared = matrix_a_inverted_shared + matrix_element_count;

	copy_triangular(matrix_size, i_column_in_a, i_column_in_b, matrix_a_shared, matrix_b_shared, matrix_a, matrix_b);

	__syncthreads();

	// compute inverse diagonal entries
	matrix_a_inverted_shared[i_column_in_a * matrix_size + i_column_in_a] =
			1.f / matrix_a_shared[i_column_in_a * matrix_size + i_column_in_a];
	matrix_b_inverted_shared[i_column_in_b * matrix_size + i_column_in_b] =
			1.f / matrix_b_shared[i_column_in_b * matrix_size + i_column_in_b];
	__syncthreads();

	invert_non_diagonal(matrix_size, i_column_in_a, i_column_in_b, matrix_a_shared, matrix_b_shared, matrix_a_inverted_shared,
	                    matrix_b_inverted_shared);

	copy_triangular(matrix_size, i_column_in_a, i_column_in_b, matrix_a, matrix_b, matrix_a_inverted_shared, matrix_b_inverted_shared);
}


template<typename scalar_t>
__device__
inline void copy_triangular_upper(
		const int matrix_size,
		const int i_column_in_a,
		const int i_column_in_b,
		scalar_t* matrix_a_dst,
		scalar_t* matrix_b_dst,
		scalar_t* matrix_a_src,
		scalar_t* matrix_b_src
) {
	const int a_nonzero_column_element_count = i_column_in_a + 1;
	// load matrix_a
	for (int i_row = 0; i_row < a_nonzero_column_element_count; i_row++) {
		matrix_a_dst[i_row * matrix_size + i_column_in_a] = matrix_a_src[i_row * matrix_size + i_column_in_a];
	}
	const int b_nonzero_column_element_count = i_column_in_b + 1;
	// load matrix_b
	for (int i_row = 0; i_row < b_nonzero_column_element_count; i_row++) {
		matrix_b_dst[i_row * matrix_size + i_column_in_b] = matrix_b_src[i_row * matrix_size + i_column_in_b];
	}
	__syncthreads();
}

template<typename scalar_t>
__device__
inline void copy_triangular_lower(
		const int matrix_size,
		const int i_column_in_a,
		const int i_column_in_b,
		scalar_t* matrix_a_dst,
		scalar_t* matrix_b_dst,
		scalar_t* matrix_a_src,
		scalar_t* matrix_b_src
) {
	const int a_nonzero_column_element_count = matrix_size - i_column_in_a;
	// load matrix_a
	for (int i_row = matrix_size - a_nonzero_column_element_count; i_row < matrix_size; i_row++) {
		matrix_a_dst[i_row * matrix_size + i_column_in_a] = matrix_a_src[i_row * matrix_size + i_column_in_a];
	}
	const int b_nonzero_column_element_count = matrix_size - i_column_in_b;
	// load matrix_b
	for (int i_row = matrix_size - b_nonzero_column_element_count; i_row < matrix_size; i_row++) {
		matrix_b_dst[i_row * matrix_size + i_column_in_b] = matrix_b_src[i_row * matrix_size + i_column_in_b];
	}
	__syncthreads();
}

template<typename scalar_t>
__device__
inline void tritri_batched_upper_non_diagonal(
		const int matrix_size,
		const int i_column_in_a,
		const int i_column_in_b,
		scalar_t* matrix_a_shared,
		scalar_t* matrix_b_shared,
		scalar_t* matrix_a_inverted_shared,
		scalar_t* matrix_b_inverted_shared
) {
	// compute upper diagonal entries of both matrices in pair
	for (int i_row = i_column_in_a - 1; i_row >= 0; i_row--) {
		float matrix_diag_entry = matrix_a_shared[i_row * matrix_size + i_row];
		float sum = 0;
		for (int k = i_row + 1; k < i_column_in_a + 1; k++) {
			sum += matrix_a_shared[i_row * matrix_size + k] * matrix_a_inverted_shared[k * matrix_size + i_column_in_a];
		}
		matrix_a_inverted_shared[i_row * matrix_size + i_column_in_a] = -sum / matrix_diag_entry;
	}
	for (int i_row = i_column_in_b - 1; i_row >= 0; i_row--) {
		float matrix_diag_entry = matrix_b_shared[i_row * matrix_size + i_row];
		float sum = 0;
		for (int k = i_row + 1; k < i_column_in_b + 1; k++) {
			sum += matrix_b_shared[i_row * matrix_size + k] * matrix_b_inverted_shared[k * matrix_size + i_column_in_b];
		}
		matrix_b_inverted_shared[i_row * matrix_size + i_column_in_b] = -sum / matrix_diag_entry;
	}
	__syncthreads();
}

template<typename scalar_t>
__device__
inline void tritri_batched_lower_non_diagonal(
		const int matrix_size,
		const int i_column_in_a,
		const int i_column_in_b,
		scalar_t* matrix_a_shared,
		scalar_t* matrix_b_shared,
		scalar_t* matrix_a_inverted_shared,
		scalar_t* matrix_b_inverted_shared
) {
	// compute lower diagonal entries of both matrices in pair
	for (int i_row = i_column_in_a + 1; i_row < matrix_size; i_row++) {
		float matrix_diag_entry = matrix_a_shared[i_row * matrix_size + i_row];
		float sum = 0;
		for (int k = i_column_in_a; k < i_row; k++) {
			sum += matrix_a_shared[i_row * matrix_size + k] * matrix_a_inverted_shared[k * matrix_size + i_column_in_a];
		}
		matrix_a_inverted_shared[i_row * matrix_size + i_column_in_a] = -sum / matrix_diag_entry;
	}
	for (int i_row = i_column_in_b + 1; i_row < matrix_size; i_row++) {
		float matrix_diag_entry = matrix_b_shared[i_row * matrix_size + i_row];
		float sum = 0;
		for (int k = i_column_in_b; k < i_row; k++) {
			sum += matrix_b_shared[i_row * matrix_size + k] * matrix_b_inverted_shared[k * matrix_size + i_column_in_b];
		}
		matrix_b_inverted_shared[i_row * matrix_size + i_column_in_b] = -sum / matrix_diag_entry;
	}
	__syncthreads();
}

template<typename scalar_t>
//square matrices are assumed
__global__
void tritri_batched_upper(
		scalar_t** matrices, int matrix_size, int matrix_pair_count_per_block,
		int jobs_per_block, int global_thread_cutoff
) {
	tritri_batched_generic(
			matrices, matrix_size, matrix_pair_count_per_block, jobs_per_block, global_thread_cutoff,
			tritri_batched_upper_non_diagonal<scalar_t>, copy_triangular_upper<scalar_t>
	);
}

template<typename scalar_t>
//square matrices are assumed
__global__
void tritri_batched_lower(
		scalar_t** matrices, int matrix_size, int matrix_pair_count_per_block,
		int jobs_per_block, int global_thread_cutoff
) {
	tritri_batched_generic(
			matrices, matrix_size, matrix_pair_count_per_block, jobs_per_block, global_thread_cutoff,
			tritri_batched_lower_non_diagonal<scalar_t>, copy_triangular_lower<scalar_t>
	);
}


template<typename scalar_t>
void trtri_batched_cuda(
		scalar_t** matrices, int matrix_count, int matrix_size,
		nnrt::core::linalg::UpLoTriangular uplo, const open3d::core::Device& device
) {
	int cuda_threads_per_thread_block = OPTIMAL_CUDA_BLOCK_THREAD_COUNT;
	int matrix_pair_count_per_thread_block = ceildiv(cuda_threads_per_thread_block, matrix_size);
	int matrix_pair_count = ceildiv(matrix_count, 2);
	int cuda_thread_block_count = ceildiv(matrix_pair_count, matrix_pair_count_per_thread_block);
	int jobs_per_block = matrix_pair_count_per_thread_block * matrix_size;
	int last_block_job_count = matrix_pair_count * matrix_size - (jobs_per_block * (cuda_thread_block_count - 1));
	int thread_count_before_cutoff = cuda_threads_per_thread_block * (cuda_thread_block_count - 1) + last_block_job_count;

	auto thread_block_memory_size = jobs_per_block * (matrix_size * 4) * sizeof(scalar_t);

	open3d::core::CUDAScopedDevice scoped_device(device);
	switch (uplo) {
		case UpLoTriangular::LOWER:
			tritri_batched_lower<<<cuda_thread_block_count, cuda_threads_per_thread_block, thread_block_memory_size, open3d::core::cuda::GetStream()>>>(
					matrices, matrix_size, matrix_pair_count_per_thread_block, jobs_per_block, thread_count_before_cutoff
			);
			break;
		case UpLoTriangular::UPPER:
			tritri_batched_upper<<<cuda_thread_block_count, cuda_threads_per_thread_block, thread_block_memory_size, open3d::core::cuda::GetStream()>>>(
					matrices, matrix_size, matrix_pair_count_per_thread_block, jobs_per_block, thread_count_before_cutoff
			);
			break;
	}
}


} // namespace

template<typename scalar_t>
void InvertTriangularBlocksCUDA_Generic(
		void* A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Dtype& data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	int effective_block_count = block_count;
	void* A_padding = nullptr;
	o3c::Tensor eye_padding;
	int padding_count = 0;
	if (block_count % 2 == 1) {
		effective_block_count += 1;
		eye_padding = o3c::Tensor::Eye(block_size, data_type, device);
		A_padding = eye_padding.GetDataPtr();
		padding_count = 1;
	}

	scalar_t** A_array_device;
	auto size_of_pointer_array = effective_block_count * sizeof(scalar_t*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	GetMatrixPointersFromContiguousArrayOfMatrices_CUDA(A_array_device, A_block_data, A_padding, block_size, block_size, block_count, padding_count,
	                                                    device);

	//TODO
	trtri_batched_cuda(A_array_device, effective_block_count, block_size, uplo, device);

	OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
}

void InvertTriangularBlocksCUDA(
		void* A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Dtype& data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		InvertTriangularBlocksCUDA_Generic<scalar_t>(
				A_block_data,
				block_size,
				block_count,
				data_type,
				device,
				uplo
		);
	});
}


} // nnrt::core::linalg::internal