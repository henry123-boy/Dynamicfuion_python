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
#include "core/linalg/InvertBlocks.h"
#include <open3d/utility/Logging.h>
namespace utility = open3d::utility;

#ifdef BUILD_CUDA_MODULE

// third-party includes
#include <open3d/core/Dtype.h>
#include <open3d/core/Device.h>
#include <open3d/utility/Parallel.h>
#include <open3d/core/CUDAUtils.h>

// local includes
#include "core/linalg/LinalgUtils.h"
#include "core/linalg/PointerAggregationForBatchOperationsCUDA.cuh"
#include "core/linalg/LinalgKernels.cuh"
#include "LapackWrapper.h"
#include "MagmaManager.h"


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
	int effective_block_count = static_cast<int>(block_count);
	void* A_padding = nullptr;
	o3c::Tensor eye_padding;
	int padding_count = 0;
	if (block_count % 2 == 1) {
		effective_block_count++;
		eye_padding = o3c::Tensor::Eye(block_size, data_type, device);
		A_padding = eye_padding.GetDataPtr();
		padding_count = 1;
	}

	scalar_t** A_array_device;
	auto size_of_pointer_array = effective_block_count * sizeof(scalar_t*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	GetMatrixPointersFromContiguousArrayOfMatricesPadded_CUDA(A_array_device, A_block_data, A_padding, block_size, block_size, block_count,
	                                                          padding_count,
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


template<typename scalar_t>
void SolveBlocksCUDA_TypeDispatched(
		void* a_block_data,
		void* b_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Device& device
) {
	auto* a_block_data_typed = static_cast<scalar_t*>(a_block_data);
	auto* b_block_data_typed = static_cast<scalar_t*>(b_block_data);

	scalar_t** A_array_device;
	scalar_t** B_array_device;
	auto size_of_pointer_array = block_count * sizeof(scalar_t*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&B_array_device, size_of_pointer_array));
	GetMatrixPointersFromContiguousArrayOfMatrices_AB_CUDA(
			A_array_device, B_array_device, a_block_data_typed, b_block_data_typed, block_size, block_size, block_count, device
	);
	cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();

	int* factorization_result;
	OPEN3D_CUDA_CHECK(cudaMalloc(&factorization_result, block_count * sizeof(int)));

	NNRT_CUSOLVER_CHECK_WITH_MULTIPLE_DINFO(
			potrf_batched_cuda<scalar_t>(
					cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
					block_size,
					A_array_device, block_size,
					factorization_result, block_count
			), "Batched portf failed in SolveBlocksCUDA", factorization_result, block_count, device
	);

	core::linalg::MagmaManager::GetInstance().SetDevice(device.GetID());
	NNRT_MAGMA_CHECK(
			potrs_batched_cuda<scalar_t>(
					magma_uplo_t::MagmaUpper,
					block_size, block_size,
					A_array_device, block_size,
					B_array_device, block_size,
					block_count,
					core::linalg::MagmaManager::GetInstance().GetDefaultQueue()
			), "Batched ports failed in SolveBlocksCUDA"
	);

	OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
	OPEN3D_CUDA_CHECK(cudaFree(B_array_device));
}


void SolveBlocksCUDA(
		void* a_block_data,
		void* b_block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		SolveBlocksCUDA_TypeDispatched<scalar_t>(
				a_block_data,
				b_block_data,
				block_size,
				block_count,
				device
		);
	});

}

#else
void InvertTriangularBlocksCUDA(
		void* A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Dtype& data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	utility::LogError("Attempting to call InvertTriangularBlocksCUDA routine when library not compiled with BUILD_CUDA_MODULE=ON");
}

void SolveBlocksCUDA(
		void* a_block_data,
		void* b_block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
) {
	utility::LogError("Attempting to call SolveBlocksCUDA routine when library not compiled with BUILD_CUDA_MODULE=ON");
}
#endif

} // nnrt::core::linalg::internal