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
#include "core/linalg/BlockSums.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::linalg {

open3d::core::Tensor ComputeBlockSums(
		int sum_count,
		const open3d::core::Tensor& blocks,
		const open3d::core::Tensor& block_sum_indices,
		int block_count
) {
	open3d::core::Tensor sums;
	core::ExecuteOnDevice(
			blocks.GetDevice(),
			[&]() {
				internal::ComputeBlockSums<open3d::core::Device::DeviceType::CPU>(sums, sum_count, blocks, block_sum_indices, block_count);
			},
			[&]() {
				NNRT_IF_CUDA(
						internal::ComputeBlockSums<open3d::core::Device::DeviceType::CUDA>(sums, sum_count, blocks, block_sum_indices, block_count);
				);
			}
	);
	return sums;
}

} // namespace nnrt::core::linalg