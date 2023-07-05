//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/5/23.
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
// third-party includes
#include <open3d/core/ParallelFor.h>
// local includes
#include "core/linalg/PreconditionDiagonalBlocks.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDevice>
void PreconditionDiagonalBlocks(
		open3d::core::Tensor& blocks,
		float dampening_factor
) {
	if (blocks.GetShape().size() != 3) {
		utility::LogError("Expecting `blocks` to have three dimensions, got dimension count: {}", blocks.GetShape().size());
	}
	int64_t block_size = blocks.GetShape(1);
	int64_t block_count = blocks.GetShape(0);
	o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks, o3c::Float32);
	o3c::Device device = blocks.GetDevice();

	int64_t block_size_squared = block_size * block_size;
	auto* block_data = blocks.GetDataPtr<float>();
	o3c::ParallelFor(
			device, block_size * block_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_index) {
				int64_t i_block = workload_index / block_size;
				int64_t i_position_in_block = workload_index % block_size;
				block_data[i_block * (block_size_squared) + i_position_in_block * block_size + i_position_in_block] += dampening_factor;
			}
	);
}

} // namespace nnrt::core::linalg::internal