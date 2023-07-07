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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>

// local includes
#include "core/linalg/BlockSums.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

static void RunBlockSumChecks(
		o3c::Tensor& sums, int sum_count, const o3c::Tensor& blocks, const o3c::Tensor& block_sum_indices, int block_count
) {
	// counters & checks
	o3c::Device device = blocks.GetDevice();
	o3c::AssertTensorDtype(sums, o3c::Float32);
	o3c::AssertTensorDtype(blocks, o3c::Float32);
	o3c::AssertTensorDtype(block_sum_indices, o3c::Int32);
	int64_t block_row_count = blocks.GetShape(1);

	int64_t max_block_count = blocks.GetShape(0);
	int64_t max_sum_count = sums.GetShape(0);
	if (blocks.NumDims() > 2) {
		int64_t block_column_count = blocks.GetShape(2);
		o3c::AssertTensorShape(sums, { max_sum_count, block_row_count, block_column_count });
		o3c::AssertTensorShape(blocks, { max_block_count, block_row_count, block_column_count });
	} else {
		o3c::AssertTensorShape(sums, { max_sum_count, block_row_count });
		o3c::AssertTensorShape(blocks, { max_block_count, block_row_count });
	}

	o3c::AssertTensorShape(block_sum_indices, { max_block_count });

	o3c::AssertTensorDevice(sums, device);
	o3c::AssertTensorDevice(block_sum_indices, device);

	if (block_count > max_block_count) {
		utility::LogError("Block count, {}, exceeds allowed maximum, {}.", block_count, max_block_count);
	}
	if (sum_count > max_sum_count) {
		utility::LogError("Sum count, {}, exceeds allowed maximum, {}.", sum_count, max_sum_count);
	}
}

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void ComputeBlockSums(
		core::AtomicTensor<TDeviceType, TElement>& sums,
		int sum_count,
		const o3c::Tensor& blocks,
		const o3c::Tensor& block_sum_indices,
		int block_count
) {
	auto sums_tensor = sums.AsTensor(false);
	RunBlockSumChecks(sums_tensor, sum_count, blocks, block_sum_indices, block_count);
	o3c::Device device = blocks.GetDevice();
	int64_t block_size = blocks.GetShape(1);
	int64_t block_stride = blocks.NumDims() == 2 ? block_size : block_size * block_size;
	auto block_sum_index_data = block_sum_indices.GetDataPtr<int32_t>();
	auto block_data = blocks.GetDataPtr<TElement>();

	o3c::ParallelFor(
			device,
			block_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_block = workload_idx / block_stride;
				int64_t i_coefficient = workload_idx % block_stride;
				int32_t i_sum = block_sum_index_data[i_block];
				sums.FetchAdd(i_sum * block_stride + i_coefficient, block_data[workload_idx]);
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void ComputeBlockSums(
		open3d::core::Tensor& sums,
		int sum_count,
		const open3d::core::Tensor& blocks,
		const open3d::core::Tensor& block_sum_indices,
		int block_count
) {
	int64_t block_size = blocks.GetShape(1);
	DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(blocks.GetDtype(), [&]() {
		core::AtomicTensor<TDeviceType, scalar_t> sums_atomic({sum_count, block_size, block_size}, blocks.GetDevice());
		ComputeBlockSums<open3d::core::Device::DeviceType::CUDA>(sums_atomic, sum_count, blocks, block_sum_indices, block_count);
		sums = sums_atomic.AsTensor(true);
	});
}

} // namespace nnrt::core::linalg::internal