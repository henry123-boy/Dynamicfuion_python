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
#include "core/linalg/LinalgKernels.cuh"


namespace utility = open3d::utility;
namespace o3c = open3d::core;

namespace nnrt::core::linalg::internal {

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