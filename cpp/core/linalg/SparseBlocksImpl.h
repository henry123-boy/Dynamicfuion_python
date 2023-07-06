//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/3/23.
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
#include "core/linalg/SparseBlocks.h"
#include "core/platform_independence/Qualifiers.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDeviceType, typename TOp>
void SparseBlocksOp_Generic(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks, const open3d::core::Tensor& coordinates,
							bool transpose, TOp&& operation) {
	// counts & checks
	int64_t matrix_row_count = matrix.GetShape(0);
	int64_t matrix_column_count = matrix.GetShape(1);
	int64_t block_count = blocks.GetShape(0);
	int64_t block_size = blocks.GetShape(1);
	int64_t block_stride = block_size * block_size;
	o3c::Device device = blocks.GetDevice();

	o3c::AssertTensorDevice(matrix, device);
	o3c::AssertTensorShape(matrix, { matrix_row_count, matrix_column_count });
	o3c::AssertTensorDtype(matrix, o3c::Float32);

	if (matrix_row_count % block_size != 0 || matrix_column_count % block_size != 0) {
		utility::LogError("Matrix row & column counts (presently, {} and {}) should be evenly divisible by block size, presently {}.",
		                  matrix_row_count, matrix_column_count);
	}

	o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks, o3c::Float32);

	o3c::AssertTensorDevice(coordinates, device);
	o3c::AssertTensorShape(coordinates, { block_count, 2 });
	o3c::AssertTensorDtype(coordinates, o3c::Int32);

	// retrieve direct memory accessors
	auto matrix_data = matrix.GetDataPtr<float>();
	auto block_data = blocks.GetDataPtr<float>();
	auto coordinate_data = coordinates.GetDataPtr<int32_t>();

	// fill matrix with blocks
	if (transpose) {
		o3c::ParallelFor(
				device,
				block_count * block_stride,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_coefficient) {
					int64_t i_block = i_block_coefficient / block_stride;
					int64_t i_coefficient_in_block = i_block_coefficient % block_stride;
					int64_t i_block_row = coordinate_data[i_block * 2 + 1]; // note order flip here, (i, j) --> (j, i)
					int64_t i_block_column = coordinate_data[i_block * 2];
					int64_t i_row = i_block_row * block_size + i_coefficient_in_block % block_size; // another in-block flip here
					int64_t i_column = i_block_column * block_size + i_coefficient_in_block / block_size;
					operation(matrix_data, i_row * matrix_column_count + i_column, block_data[i_block_coefficient]);
					// matrix_data[i_row * matrix_column_count + i_column] = block_data[i_block_coefficient];
				}
		);
	} else {
		o3c::ParallelFor(
				device,
				block_count * block_stride,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_coefficient) {
					int64_t i_block = i_block_coefficient / block_stride;
					int64_t i_coefficient_in_block = i_block_coefficient % block_stride;
					int64_t i_block_row = coordinate_data[i_block * 2];
					int64_t i_block_column = coordinate_data[i_block * 2 + 1];
					int64_t i_row = i_block_row * block_size + i_coefficient_in_block / block_size;
					int64_t i_column = i_block_column * block_size + i_coefficient_in_block % block_size;
					operation(matrix_data, i_row * matrix_column_count + i_column, block_data[i_block_coefficient]);
					// matrix_data[i_row * matrix_column_count + i_column] = block_data[i_block_coefficient];
				}

		);
	}
}

template<open3d::core::Device::DeviceType TDeviceType>
void FillInSparseBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks, const open3d::core::Tensor& coordinates, bool transpose){
	SparseBlocksOp_Generic<TDeviceType>(
			matrix, blocks, coordinates, transpose,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(float* matrix_data, int64_t datum_index, float operand){
				matrix_data[datum_index] = operand;
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void AddSparseBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks, const open3d::core::Tensor& coordinates, bool transpose){
	SparseBlocksOp_Generic<TDeviceType>(
			matrix, blocks, coordinates, transpose,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(float* matrix_data, int64_t datum_index, float operand){
				matrix_data[datum_index] += operand;
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void SubtractSparseBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks, const open3d::core::Tensor& coordinates, bool transpose){
	SparseBlocksOp_Generic<TDeviceType>(
			matrix, blocks, coordinates, transpose,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(float* matrix_data, int64_t datum_index, float operand){
				matrix_data[datum_index] -= operand;
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
open3d::core::Tensor GetSparseBlocks(const open3d::core::Tensor& matrix, int block_size, const open3d::core::Tensor& coordinates) {
	// counts & checks
	int64_t matrix_row_count = matrix.GetShape(0);
	int64_t matrix_column_count = matrix.GetShape(1);
	int64_t block_count = coordinates.GetShape(0);
	int64_t block_stride = block_size * block_size;
	o3c::Device device = coordinates.GetDevice();

	o3c::AssertTensorDevice(matrix, device);
	o3c::AssertTensorShape(matrix, { matrix_row_count, matrix_column_count });
	o3c::AssertTensorDtype(matrix, o3c::Float32);

	o3c::AssertTensorShape(coordinates, { block_count, 2 });
	o3c::AssertTensorDtype(coordinates, o3c::Int32);

	// output init & accessors
	o3c::Tensor blocks({block_count, block_size, block_size}, o3c::Float32, device);
	auto matrix_data = matrix.GetDataPtr<float>();
	auto block_data = blocks.GetDataPtr<float>();
	auto coordinate_data = coordinates.GetDataPtr<int32_t>();

	// fill blocks from matrix
	o3c::ParallelFor(
			device,
			block_count * block_stride,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_block_coefficient) {
				int64_t i_block = i_block_coefficient / block_stride;
				int64_t i_coefficient_in_block = i_block_coefficient % block_stride;
				int64_t i_block_row = coordinate_data[i_block * 2];
				int64_t i_block_column = coordinate_data[i_block * 2 + 1];
				int64_t i_row = i_block_row * block_size + i_coefficient_in_block / block_size;
				int64_t i_column = i_block_column * block_size + i_coefficient_in_block % block_size;
				block_data[i_block_coefficient] = matrix_data[i_row * matrix_column_count + i_column];
			}
	);
	return blocks;
}

} // namespace nnrt::core::linalg::internal