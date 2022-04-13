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

#include "core/linalg/LinalgUtils.h"
#include "core/kernel/Matmul3D.h"
#include "core/linalg/BlasWrapper.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace nnrt::core::kernel {

template<>
void Matmul3D<open3d::core::Device::DeviceType::CUDA>(const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
													  int64_t batch_size, open3d::core::Dtype dtype) {

	cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();

	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
		float alpha = 1, beta = 0;
		const float* A_data_cast = static_cast<const float*>(A);
		const float* B_data_cast = static_cast<const float*>(B);
		float* C_data_cast = static_cast<float*>(C);
		NNRT_CUBLAS_CHECK(
				gemm_batched_cuda<float>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				                         m, n, k,
				                         &alpha,
				                         &A_data_cast, m,
				                         &B_data_cast, k,
				                         &beta,
				                         &C_data_cast, m,
				                         batch_size),
				"cuda batched gemm failed");
	});






}

} // nnrt::core::kernel