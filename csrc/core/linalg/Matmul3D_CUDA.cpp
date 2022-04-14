//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/12/22.
//  Copyright (c) 2022 Gregory Kramida
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

#include <open3d/core/Tensor.h>
#include <open3d/utility/Logging.h>
#include <open3d/core/Blob.h>
#include <open3d/core/CUDAUtils.h>

#include "core/linalg/LinalgUtils.h"
#include "Matmul3D.h"
#include "core/linalg/BlasWrapper.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace nnrt::core::linalg {

template<>
void Matmul3D<open3d::core::Device::DeviceType::CUDA>(const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
                                                      int64_t batch_size, open3d::core::Dtype dtype) {

	cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();

	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
		scalar_t alpha = 1, beta = 0;

		auto A_data = static_cast<const scalar_t*>(A);
		auto B_data = static_cast<const scalar_t*>(B);
		auto C_data = static_cast<scalar_t*>(C);

		const scalar_t* A_array[batch_size];
		const scalar_t* B_array[batch_size];
		scalar_t* C_array[batch_size];

		get_matrix_pointers_from_contiguous_array_of_matrices<scalar_t>(A_array, B_array, C_array, A, B, C, m, k, n, batch_size);

		const scalar_t** A_array_device;
		const scalar_t** B_array_device;
		scalar_t** C_array_device;

		auto size_of_pointer_array =  batch_size * sizeof(scalar_t*);

		OPEN3D_CUDA_CHECK(cudaMalloc(&A_array_device, size_of_pointer_array));
		OPEN3D_CUDA_CHECK(cudaMalloc(&B_array_device, size_of_pointer_array));
		OPEN3D_CUDA_CHECK(cudaMalloc(&C_array_device, size_of_pointer_array));

		OPEN3D_CUDA_CHECK(cudaMemcpy(A_array_device, A_array, size_of_pointer_array, cudaMemcpyHostToDevice));
		OPEN3D_CUDA_CHECK(cudaMemcpy(B_array_device, B_array, size_of_pointer_array, cudaMemcpyHostToDevice));
		OPEN3D_CUDA_CHECK(cudaMemcpy(C_array_device, C_array, size_of_pointer_array, cudaMemcpyHostToDevice));

		NNRT_CUBLAS_CHECK(
				gemm_batched_cuda<scalar_t>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				                            n, m, k,
				                            &alpha,
				                            B_array_device, n,
				                            A_array_device, k,
				                            &beta,
				                            C_array_device, n,
				                            batch_size),
				"cuda batched gemm failed");
		OPEN3D_CUDA_CHECK(cudaFree(A_array_device));
		OPEN3D_CUDA_CHECK(cudaFree(B_array_device));
		OPEN3D_CUDA_CHECK(cudaFree(C_array_device));
	});


}

} // nnrt::core::linalg