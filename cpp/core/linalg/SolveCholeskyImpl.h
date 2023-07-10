//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/7/23.
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
// local includes
#include "core/linalg/SolveCholesky.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/LinalgUtils.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDeviceType, typename scalar_t, bool THasVectorSolution>
void SolveCholesky_Generic(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
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

	// accessors & output prep

#ifdef __CUDACC__ //__DEBUG
	// #if true
		scalar_t* X_transposed_data;
		const scalar_t* B_data;
		if(THasVectorSolution){
			X = B.Clone();
		} else {
			o3c::Tensor X_transposed({ nrhs, n }, dtype, device);
			X = o3c::Tensor({ n, nrhs }, dtype, device);
			X_transposed_data = X_transposed.GetDataPtr<scalar_t>();
			B_data = B.GetDataPtr<scalar_t>();
		}

		// will be "cloned" during the transpose
		o3c::Tensor A_factorized({ n, n }, dtype, device);
		auto A_data = A.GetDataPtr<scalar_t>();
#else
	X = B.Clone();
	auto A_factorized = A.Clone();
#endif
	auto A_factorized_data = A_factorized.GetDataPtr<scalar_t>();
	auto X_data = X.GetDataPtr<scalar_t>();

#ifdef __CUDACC__
	cublasHandle_t cublas_handle = CuBLASContext::GetInstance()->GetHandle();

	scalar_t alpha = 1.0, beta = 0.;
	NNRT_CUBLAS_CHECK(
			geam_cuda(cublas_handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N,
					  n, n, &alpha, A_data, n, &beta, A_data, n,
					  A_factorized_data, n), "geam failed in SolveCholesky on CUDA"
	);
	cusolverDnHandle_t cusolver_dn_handle = CuSolverContext::GetInstance()->GetHandle();
	int* device_info = nullptr;
	cusolverStatus_t status = potrf_cuda(cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, A_factorized_data, n, &device_info);
	if (device_info != nullptr) {
		NNRT_CUSOLVER_CHECK_WITH_DINFO(status, "potrf failed in SolveCholesky on CUDA", device_info, device);
	} else {
		NNRT_CUSOLVER_CHECK(status, "portf buffer allocation failed in SolveCholesky on CUDA");
	}
	if (THasVectorSolution) {
		NNRT_CUSOLVER_CHECK_WITH_DINFO(
				potrs_cuda(cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, nrhs, A_factorized_data, n, X_data, n, device_info),
				"potrf failed in SolveCholesky on CUDA", device_info, device
		);
		OPEN3D_CUDA_CHECK(cudaFree(device_info));

	} else {
		// extra transpose (B --> B.T) is required if B is not one-dimensional.
		NNRT_CUBLAS_CHECK(
				geam_cuda(cublas_handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N,
						  nrhs, n, &alpha, B_data, n, &beta, B_data, n,
						  X_transposed_data, n), "geam failed in SolveCholesky on CUDA"
		);
		NNRT_CUSOLVER_CHECK_WITH_DINFO(
				potrs_cuda(cusolver_dn_handle, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, nrhs, A_factorized_data, n, X_transposed_data, n, device_info),
				"potrf failed in SolveCholesky on CUDA", device_info, device
		);
		NNRT_CUBLAS_CHECK(
				geam_cuda(cublas_handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N,
						  n, nrhs, &alpha, X_transposed_data, n, &beta, X_data, n,
						  X_data, n), "geam failed in SolveCholesky on CUDA"
		);
	}
#else
	NNRT_LAPACK_CHECK(potrf_cpu(LAPACK_ROW_MAJOR, 'l', n, A_factorized_data, n), "potrf failed in SolveCholesky on CPU");
	NNRT_LAPACK_CHECK(potrs_cpu(LAPACK_ROW_MAJOR, 'l', n, nrhs, A_factorized_data, n, X_data, nrhs), "potrs failed in SolveCholesky on CPU");
#endif
	//__DEBUG
	auto A_factorized_CPU = A_factorized.To(o3c::Device("CPU:0"));
	auto X_CPU = X.To(o3c::Device("CPU:0"));
#ifdef __CUDACC__
	// auto X_transposed_CPU = X_transposed.To(o3c::Device("CPU:0"));
	// X = X_transposed;
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void SolveCholesky(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
	DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
			A.GetDtype(),
			[&] {
				bool has_vector_solution = B.NumDims() == 1 || B.GetShape(1) == 1LL;
				if (has_vector_solution) {
					SolveCholesky_Generic<TDeviceType, scalar_t, true>(X, A, B);
				} else {
					SolveCholesky_Generic<TDeviceType, scalar_t, false>(X, A, B);
				}
			}
	);
}

} // namespace nnrt::core::linalg::internal
