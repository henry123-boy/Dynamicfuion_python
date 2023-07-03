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
#include "open3d/core/ParallelFor.h"
// local includes
#include "core/linalg/DiagonalBlocks.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;
namespace nnrt::core::linalg::internal {
template<open3d::core::Device::DeviceType TDeviceType>
void FillInDiagonalBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks) {
	auto device = blocks.GetDevice();
	int64_t matrix_size = matrix.GetShape(0);
	int64_t block_count = blocks.GetShape(0);
	int64_t block_size = blocks.GetShape(1);

	o3c::AssertTensorDevice(matrix, device);
	o3c::AssertTensorDtype(matrix, o3c::Float32);
	o3c::AssertTensorShape(matrix, { matrix_size, matrix_size });

	o3c::AssertTensorDtype(blocks, o3c::Float32);
	o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });

	if (block_count * block_size != matrix_size) {
		open3d::utility::LogError("block_count X block_size, currently {} x {} = {}, must equal matrix_size, currently {}.",
		                          block_count, block_size, block_count * block_size, matrix_size);
	}

	auto block_stride = block_size * block_size;
	auto block_element_count = block_count * block_stride;

	auto matrix_data = matrix.GetDataPtr<float>();
	auto block_data = blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device,
			block_element_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_element) {
				int64_t i_block = i_block_element / block_stride;
				int64_t i_element_in_block = i_block_element % block_stride;
				int64_t i_row = i_block * block_size + i_element_in_block / block_size;
				int64_t i_column = i_block * block_size + i_element_in_block % block_size;
				matrix_data[i_row * matrix_size + i_column] = block_data[i_block_element];
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
open3d::core::Tensor GetDiagonalBlocks(const open3d::core::Tensor& matrix, int block_size) {
	auto device = matrix.GetDevice();
	int64_t matrix_size = matrix.GetShape(0);
	if (matrix_size % block_size != 0) {
		open3d::utility::LogError("Matrix size, currently {}, should be evenly divisible by block_size, {}, which it is not.",
		                          matrix_size, block_size);
	}

	int64_t block_count = matrix_size / block_size;

	o3c::AssertTensorDtype(matrix, o3c::Float32);
	o3c::AssertTensorShape(matrix, { matrix_size, matrix_size });

	o3c::Tensor blocks({block_count, block_size, block_size}, o3c::Float32, device);

	auto block_stride = block_size * block_size;
	auto block_element_count = block_count * block_stride;

	auto matrix_data = matrix.GetDataPtr<float>();
	auto block_data = blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device,
			block_element_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_element) {
				int64_t i_block = i_block_element / block_stride;
				int64_t i_element_in_block = i_block_element % block_stride;
				int64_t i_row = i_block * block_size + i_element_in_block / block_size;
				int64_t i_column = i_block * block_size + i_element_in_block % block_size;
				block_data[i_block_element] = matrix_data[i_row * matrix_size + i_column];
			}
	);
	return blocks;
}

} // namespace nnrt::core::linalg::internal