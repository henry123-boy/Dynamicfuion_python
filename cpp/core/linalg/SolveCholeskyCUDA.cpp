//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/10/23.
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
#ifdef BUILD_CUDA_MODULE

// local includes
#include "core/linalg/SolveCholesky.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/LinalgUtils.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<typename scalar_t>
static void SolveCholeskyCUDA_Generic(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
	// counters & checks
	o3c::Device device = A.GetDevice();
	o3c::Dtype dtype = A.GetDtype();
	const int64_t n = A.GetShape(0);
	int64_t nrhs;
	if (B.NumDims() == 2) {
		nrhs = B.GetShape(1);
		o3c::AssertTensorShape(B, { n, nrhs });
	} else {
		nrhs = 1;
		o3c::AssertTensorShape(B, { n });
	}

	o3c::AssertTensorShape(A, { n, n });

	o3c::AssertTensorDtype(B, dtype);
	o3c::AssertTensorDevice(B, device);

	X = B.Clone();

	auto A_factorized = A.Clone();
	auto A_factorized_data = A_factorized.GetDataPtr<scalar_t>();

	// auto A_data = A.GetDataPtr<scalar_t>();
	auto X_data = X.GetDataPtr<scalar_t>();

	cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();
	int* device_info = nullptr;
	cusolverStatus_t status = potrf_cuda(cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, A_factorized_data, n, &device_info);
	if (device_info != nullptr) {
		NNRT_CUSOLVER_CHECK_WITH_DINFO(status, "potrf failed in SolveCholesky on CUDA", device_info, device);
	} else {
		NNRT_CUSOLVER_CHECK(status, "portf buffer allocation failed in SolveCholesky on CUDA");
	}

	auto A_factorized_CPU = A_factorized.To(o3c::Device("CPU:0"));


	cublasHandle_t cublas_handle = CuBLASContext::GetInstance()->GetHandle();

	scalar_t alpha = 1.0;
	NNRT_CUBLAS_CHECK(
			trsm_cuda<scalar_t>(cublas_handle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
			                    cublasOperation_t::CUBLAS_OP_N, cublasDiagType_t::CUBLAS_DIAG_NON_UNIT, nrhs,
			                    n, &alpha, A_factorized_data, n, X_data, nrhs),
			"trsm failed in SolveCholeskyCUDA"
	);
	NNRT_CUBLAS_CHECK(
			trsm_cuda<scalar_t>(cublas_handle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
			                    cublasOperation_t::CUBLAS_OP_T, cublasDiagType_t::CUBLAS_DIAG_NON_UNIT, nrhs,
			                    n, &alpha, A_factorized_data, n, X_data, nrhs),
			"trsm failed in SolveCholeskyCUDA"
	);

	OPEN3D_CUDA_CHECK(cudaFree(device_info));
}

void SolveCholeskyCUDA(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
	DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
			A.GetDtype(),
			[&] {
				SolveCholeskyCUDA_Generic<scalar_t>(X, A, B);
			}
	);
}

#else
void SolveCholeskyCUDA(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
	utility::LogError("Attempting to call SolveCholeskyCUDA routine when library not compiled with BUILD_CUDA_MODULE=ON");
}
#endif

} // namespace nnrt::core::linalg::internal