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

#ifdef BUILD_CUDA_MODULE

// local includes
#include "core/linalg/InvertBlocks.h"
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/PointerAggregationForBatchOperations.h"
#include "MagmaManager.h"


namespace utility = open3d::utility;
namespace nnrt::core::linalg::internal {


template<typename scalar_t>
void InvertTriangularBlocksCUDA_Generic(
		const void* A_block_data,
		void* inv_A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	const scalar_t* A_array[block_count];
	scalar_t* inv_A_array[block_count];
	GetMatrixPointersFromContiguousArrayOfMatrices_constA_B(A_array, inv_A_array, A_block_data, inv_A_block_data, block_size, block_size, block_count);
	scalar_t** A_array_device;
	scalar_t** inv_A_array_device;

	auto size_of_pointer_array = block_count * sizeof(scalar_t*);

	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMemcpy(A_array_device, A_array, size_of_pointer_array, cudaMemcpyHostToDevice));
	OPEN3D_CUDA_CHECK(cudaMalloc(&inv_A_array_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMemcpy(inv_A_array_device, inv_A_array, size_of_pointer_array, cudaMemcpyHostToDevice));

	MagmaManager::GetInstance().SetDevice(device.GetID());

	trtri_batched_cuda<scalar_t>(
			// triangular upper/lower variant "flips" because matrix layout also flips (row->col major) during the computation
			uplo == UpLoTriangular::UPPER ? magma_uplo_t::MagmaLower : magma_uplo_t::MagmaUpper,
			magma_diag_t::MagmaNonUnit, block_size * block_count, A_array_device, block_size, inv_A_array_device, 0, 1,
			MagmaManager::GetInstance().GetDefaultQueue()
	);


	OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
	OPEN3D_CUDA_CHECK(cudaFree(inv_A_array_device));
}

void InvertTriangularBlocksCUDA(
		const void* A_block_data,
		void* inv_A_block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		InvertTriangularBlocksCUDA_Generic<scalar_t>(
				A_block_data,
				inv_A_block_data,
				block_size,
				block_count,
				device,
				uplo
		);
	});
}


} // nnrt::core::linalg::internal
#else
namespace nnrt::core::linalg::internal{


void InvertBlocksCUDA(
		void* block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	utility::LogError("Attempting to call InvertBlocksCUDA routine when library not compiled with BUILD_CUDA_MODULE=ON");
}

} // nnrt::core::linalg::internal
#endif