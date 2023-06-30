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
#include <open3d/core/CUDAUtils.h>


// local includes
#include "core/linalg/FactorizeBlocksCholesky.h"

#ifdef BUILD_CUDA_MODULE

#include "core/linalg/LinalgUtils.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/PointerAggregationForBatchOperationsCPU.h"


namespace utility = open3d::utility;
namespace nnrt::core::linalg::internal {

template<typename scalar_t>
void FactorizeBlocksCholeskyCUDA_Generic(
		void* block_data,
		int64_t block_size,
		int64_t block_count,
		UpLoTriangular uplo
) {
	scalar_t* A_array[block_count];
	GetMatrixPointersFromContiguousArrayOfMatrices_CPU(A_array, block_data, block_size, block_size, block_count);
	scalar_t** A_array_device;

	auto size_of_pointer_array = block_count * sizeof(scalar_t*);

	OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMemcpy(A_array_device, A_array, size_of_pointer_array, cudaMemcpyHostToDevice));

	cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();

	int* info_array;
	OPEN3D_CUDA_CHECK(cudaMalloc(&info_array, block_count * sizeof(int)));

	NNRT_CUSOLVER_CHECK(
			potrf_batched_cuda<scalar_t>(
					cusolver_dn_handle, uplo == UpLoTriangular::LOWER ?
					                    cublasFillMode_t::CUBLAS_FILL_MODE_UPPER : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
					block_size, A_array_device,
					block_size, info_array, block_count
			), "Batched portf failed in SolveBlockDiagonalCUDACholesky"
	);

	OPEN3D_CUDA_CHECK(cudaFree(info_array));
	OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
}

void FactorizeBlocksCholeskyCUDA(
		void* block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device,
		UpLoTriangular uplo
) {
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(data_type, [&]() {
		FactorizeBlocksCholeskyCUDA_Generic<scalar_t>(
				block_data,
				block_size,
				block_count,
				uplo
		);
	});
}

} // namespace nnrt::core::linalg::internal
#else
namespace nnrt::core::linalg::internal{
void FactorizeBlocksCholeskyCUDA(
		void* block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
) {
	utility::LogError("Attempting to call FactorizeBlocksCholeskyCUDA routine when library not compiled with BUILD_CUDA_MODULE=ON");
}
}
#endif