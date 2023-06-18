//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/23.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
// local includes
#include "core/linalg/BlockSparseArrowheadMatrix.h"

namespace nnrt::core::linalg {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
        FactorizeBlockSparseArrowheadCholesky_Upper(const BlockSparseArrowheadMatrix& A);

void SolveBlockSparseArrowheadCholesky(open3d::core::Tensor& X, const BlockSparseArrowheadMatrix& A, const open3d::core::Tensor& B);


namespace internal {
void FactorizeBlockSparseCholeskyCorner(
		open3d::core::Tensor& factorized_upper_dense_corner,
		const open3d::core::Tensor& factorized_upper_blocks,
		const BlockSparseArrowheadMatrix& A
);

template<open3d::core::Device::DeviceType TDeviceType>
void FactorizeBlockSparseCholeskyCorner(
		open3d::core::Tensor& factorized_upper_dense_corner,
		const open3d::core::Tensor& factorized_upper_blocks,
		const BlockSparseArrowheadMatrix& A
);

}

} // namespace nnrt::core::linalg