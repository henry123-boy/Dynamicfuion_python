//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/30/23.
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

// local includes
#include "core/linalg/BlockSumsImpl.h"

namespace nnrt::core::linalg::internal {
template
void ComputeBlockSums<open3d::core::Device::DeviceType::CUDA, float>(
		core::AtomicTensor<open3d::core::Device::DeviceType::CUDA, float>& sums,
		int sum_count,
		const o3c::Tensor& blocks,
		const o3c::Tensor& block_sum_indices,
		int block_count
);
template
void ComputeBlockSums<open3d::core::Device::DeviceType::CUDA, double>(
		core::AtomicTensor<open3d::core::Device::DeviceType::CUDA, double>& sums,
		int sum_count,
		const o3c::Tensor& blocks,
		const o3c::Tensor& block_sum_indices,
		int block_count
);


} // namespace nnrt::core::linalg::internal