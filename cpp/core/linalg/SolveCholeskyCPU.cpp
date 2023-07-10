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
// local includes
#include "core/linalg/SolveCholesky.h"
#include "core/linalg/BlasWrapper.h"
#include "core/linalg/LapackWrapper.h"
#include "core/linalg/LinalgUtils.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<typename scalar_t>
static void SolveCholeskyCPU_Generic(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
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
	X = B.Clone();
	auto X_data = X.GetDataPtr<scalar_t>();

	auto A_factorized = A.Clone();
	auto A_factorized_data = A_factorized.GetDataPtr<scalar_t>();

	NNRT_LAPACK_CHECK(potrf_cpu(LAPACK_COL_MAJOR, 'U', n, A_factorized_data, n), "potrf failed in SolveCholesky on CPU");
	//solve LY = B
	trsm_cpu<scalar_t>(
			CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
			nrhs, n,
			static_cast<scalar_t>(1),
			A_factorized_data, n,
			X_data, nrhs // out: Y
	);
	//solve LX = B
	trsm_cpu<scalar_t>(
			CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit,
			nrhs, n,
			static_cast<scalar_t>(1),
			A_factorized_data, n,
			X_data, nrhs // out: X
	);
	// NNRT_LAPACK_CHECK(potrs_cpu(LAPACK_COL_MAJOR, 'l', n, nrhs, A_factorized_data, n, X_data, nrhs), "potrs failed in SolveCholesky on CPU");
}

void SolveCholeskyCPU(open3d::core::Tensor& X, const open3d::core::Tensor& A, const open3d::core::Tensor& B) {
	DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
			A.GetDtype(),
			[&] {
				SolveCholeskyCPU_Generic<scalar_t>(X, A, B);
			}
	);
}

} // namespace nnrt::core::linalg::internal