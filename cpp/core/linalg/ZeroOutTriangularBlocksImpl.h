//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/21/23.
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
#include "core/linalg/ZeroOutTriangularBlocks.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDeviceType>
void ZeroOutTriangularBlocks(open3d::core::Tensor& blocks, nnrt::core::linalg::UpLoTriangular up_lo_triangular) {
	int64_t block_count;
	int64_t block_size = blocks.GetShape(1);
	auto device = blocks.GetDevice();
	if (blocks.GetShape().size() == 3) {
		block_count = blocks.GetShape(0);
		o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });
	} else {
		o3c::AssertTensorShape(blocks, { block_size, block_size });
		block_count = 1;
	}
	o3c::AssertTensorDtype(blocks, o3c::Float32);
	if(!blocks.IsContiguous()){
		utility::LogError("ZeroOutTriangularBlocks only works on contiguous block arrays!");
	}
	int64_t block_stride = block_size * block_size;
	int64_t element_count = block_count * block_size * block_size;
	auto block_data = blocks.GetDataPtr<float>();
	if (up_lo_triangular == UpLoTriangular::LOWER) {
		o3c::ParallelFor(
				device,
				element_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int64_t i_block = workload_idx / block_stride;
					int64_t i_element_in_block = workload_idx % block_stride;
					int64_t i_block_row = i_element_in_block / block_size;
					int64_t i_block_column = i_element_in_block % block_size;
					if (i_block_row > i_block_column) { // blocks below the diagonal
						block_data[i_block * block_stride + i_block_row * block_size + i_block_column] = 0.0;
					}
				}
		);
	}
	if (up_lo_triangular == UpLoTriangular::UPPER) {
		o3c::ParallelFor(
				device,
				element_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					int64_t i_block = workload_idx / block_stride;
					int64_t i_element_in_block = workload_idx % block_stride;
					int64_t i_block_row = i_element_in_block / block_size;
					int64_t i_block_column = i_element_in_block % block_size;
					if (i_block_row < i_block_column) { // blocks above the diagonal
						block_data[i_block * block_stride + i_block_row * block_size + i_block_column] = 0.0;
					}
				}
		);
	}
}

} // namespace nnrt::core::linalg::internal