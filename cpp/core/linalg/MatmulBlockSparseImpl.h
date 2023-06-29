//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/23.
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
#include "core/linalg/MatmulBlockSparse.h"
#include "core/platform_independence/Qualifiers.h"
#include "LinalgUtils.h"
#include "core/linalg/BlasWrapper.h"
#include "core/platform_independence/Atomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

template<open3d::core::Device::DeviceType TDeviceType>
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates,
		bool padded
) {
	// counters and checks
	o3c::Device device = blocks_b.GetDevice();
	o3c::AssertTensorDtype(blocks_b, o3c::Float32);
	int64_t block_count = blocks_b.GetShape(0);
	int64_t block_size = blocks_b.GetShape(1);
	o3c::AssertTensorShape(blocks_b, { block_count, block_size, block_size });

	o3c::AssertTensorShape(blocks_b_coordinates, { block_count, 2 });
	o3c::AssertTensorDtype(blocks_b_coordinates, o3c::Int32);
	o3c::AssertTensorDevice(blocks_b_coordinates, device);

	int64_t a_block_count = blocks_a.GetShape(0);
	o3c::AssertTensorShape(blocks_a, { a_block_count, block_size, block_size });
	o3c::AssertTensorDtype(blocks_a, o3c::Float32);
	o3c::AssertTensorDevice(blocks_a, device);

	// prep inputs
	auto blocks_b_data = blocks_b.GetDataPtr<float>();
	auto blocks_b_coordinate_data = blocks_b_coordinates.GetDataPtr<int32_t>();
	auto blocks_a_data = blocks_a.GetDataPtr<float>();

	// prep outputs
	o3c::Tensor product_blocks = o3c::Tensor(blocks_b.GetShape(), o3c::Float32, device);
	auto blocks_c_data = product_blocks.GetDataPtr<float>();

	// loop over blocks and assign multiplication block triplets
#ifdef __CUDACC__
	const float** a_blocks_device;
	const float** b_blocks_device;
	float** c_blocks_device;
	auto size_of_pointer_array =  block_count * sizeof(float*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&a_blocks_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&b_blocks_device, size_of_pointer_array));
	OPEN3D_CUDA_CHECK(cudaMalloc(&c_blocks_device, size_of_pointer_array));
#else
	const float* a_blocks_device[block_count];
	const float* b_blocks_device[block_count];
	float* c_blocks_device[block_count];
#endif
	o3c::Tensor padding_block = o3c::Tensor::Zeros({block_size, block_size}, o3c::Float32, device);
	float* padding_data = padding_block.GetDataPtr<float>();


	int64_t block_stride = block_size * block_size;
	o3c::Tensor mask({block_count}, o3c::Bool, device);
	auto mask_data = mask.GetDataPtr<bool>();
	o3c::ParallelFor(
			device,
			block_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_input_block) {
				int i_row = blocks_b_coordinate_data[i_input_block * 2];
				if (i_row < a_block_count) {
					a_blocks_device[i_input_block] = blocks_a_data + i_row * block_stride;
					b_blocks_device[i_input_block] = blocks_b_data + i_input_block * block_stride;
					mask_data[i_input_block] = true;
				} else {
					a_blocks_device[i_input_block] = padding_data;
					b_blocks_device[i_input_block] = padding_data;
					mask_data[i_input_block] = false;
				}
				c_blocks_device[i_input_block] = blocks_c_data + i_input_block * block_stride;
			}
	);
	auto block_size_int32 = static_cast<int32_t>(block_size);
	float alpha = 1, beta = 0;

#ifdef __CUDACC__
	cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
	NNRT_CUBLAS_CHECK(
			// Note: A<->B matrix order flippage is due to difference in layout -- gemm_batched_cuda assumes column-major,
			// while open3d::core::Tensors are row-major
			gemm_batched_cuda<float>(
					handle, CUBLAS_OP_N, CUBLAS_OP_N,
					block_size_int32, block_size_int32, block_size_int32,
					&alpha,
					b_blocks_device, block_size_int32,
					a_blocks_device, block_size_int32,
					&beta,
					c_blocks_device, block_size_int32,
					block_count
			),
			"cuda batched gemm failed"
	);
#else
	gemm_batched_cpu<float>(
			CblasRowMajor, CblasNoTrans, CblasNoTrans,
			block_size_int32, block_size_int32, block_size_int32,
			alpha,
			a_blocks_device, block_size_int32,
			b_blocks_device, block_size_int32,
			beta,
			c_blocks_device, block_size_int32,
			block_count
	);
#endif

#ifdef __CUDACC__
	OPEN3D_CUDA_CHECK(cudaFree(a_blocks_device));
	OPEN3D_CUDA_CHECK(cudaFree(b_blocks_device));
	OPEN3D_CUDA_CHECK(cudaFree(c_blocks_device));
#endif
	if (padded) {
		o3c::TensorKey mask_key = o3c::TensorKey::IndexTensor(mask);
		o3c::Tensor filtered_product_blocks = product_blocks.GetItem(mask_key);
		o3c::Tensor filtered_coordinates = blocks_b_coordinates.GetItem(mask_key);
		return std::make_tuple(filtered_product_blocks, filtered_coordinates);
	} else {
		return std::make_tuple(product_blocks, o3c::Tensor());
	}
}

} // namespace nnrt::core::linalg::internal