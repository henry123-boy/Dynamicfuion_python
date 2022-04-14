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
#include "Matmul3D.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LinalgUtils.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace nnrt::core::linalg {

template<>
void Matmul3D<open3d::core::Device::DeviceType::CPU>(const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
                                                      int64_t batch_size, open3d::core::Dtype dtype) {

	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
		scalar_t
		alpha = 1, beta = 0;

		const scalar_t* A_array[batch_size];
		const scalar_t* B_array[batch_size];
		scalar_t* C_array[batch_size];

		get_matrix_pointers_from_contiguous_array_of_matrices<scalar_t>(A_array, B_array, C_array, A, B, C, m, k, n, batch_size);
		gemm_batched_cpu<scalar_t>(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
		                           A_array, k, B_array, n, beta, C_array, n, batch_size);
	});

}

} // nnrt::core::linalg