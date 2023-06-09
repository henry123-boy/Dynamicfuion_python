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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>

// local includes
#include "core/linalg/BlockSparseArrowheadMatrix.h"
namespace nnrt::alignment::functional::kernel {

void ArapSparseHessianApproximation(
		nnrt::core::linalg::BlockSparseArrowheadMatrix& arap_hessian_approximation,

		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& condensed_edge_jacobians,
		int64_t first_layer_node_count,
		int64_t node_count
);

template<open3d::core::Device::DeviceType TDeviceType>
void ArapSparseHessianApproximation(
		core::linalg::BlockSparseArrowheadMatrix& arap_hessian_approximation,

		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& condensed_edge_jacobians,
		int64_t first_layer_node_count,
		int64_t node_count
);

} // namespace nnrt::alignment::functional::kernel