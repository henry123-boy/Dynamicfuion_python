//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/4/23.
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
#include <open3d/core/Tensor.h>

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg{


template<typename FSolveOnCPU, typename FSolveOnCUDA>
void SolveBlockDiagonal_Generic(
        open3d::core::Tensor& X,
        const open3d::core::Tensor& A_blocks,
        const open3d::core::Tensor& B,
        FSolveOnCPU&& solve_on_cpu,
        FSolveOnCUDA&& solve_on_cuda
) {
    o3c::AssertTensorDtypes(A_blocks, { o3c::Float32, o3c::Float64 });
    const o3c::Device device = A_blocks.GetDevice();
    const o3c::Dtype data_type = A_blocks.GetDtype();

    o3c::AssertTensorDevice(B, device);
    o3c::AssertTensorDtype(B, data_type);

    o3c::SizeVector A_blocks_shape = A_blocks.GetShape();
    o3c::SizeVector B_shape = B.GetShape();
    if (A_blocks_shape.size() != 3) {
        utility::LogError("Tensor A_blocks must have three dimensions, got {}.", A_blocks_shape.size());
    }
    if (A_blocks_shape[1] != A_blocks_shape[2]) {
        utility::LogError("Tensor A_blocks must consist of square blocks, "
                          "i.e. dimensions at indices 1 and 2 must be equal. "
                          "Got dimensions: {} and {}..", A_blocks_shape[1], A_blocks_shape[2]);
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Expected Tensor B to have one or two dimensions (to be a vector or a matrix), "
                "but got a tensor with {} dimensions.",
                B_shape.size());
    }
    int64_t result_row_count = B_shape[0];
    if (result_row_count != A_blocks_shape[1] * A_blocks_shape[0]) {
        utility::LogError("Tensor B's row count (dimension 0) should match [block count x block rows] of A "
                          "(dimension 0 x dimension 1). Got {} rows for B and {} rows for each of the {} blocks in A.",
                          B_shape[0], A_blocks_shape[1], A_blocks_shape[0]);
    }
    int64_t block_row_count = A_blocks_shape[1];
    int64_t block_count = A_blocks_shape[0];
    int64_t result_column_count = B_shape.size() == 2 ? B_shape[1] : 1;

    if (block_row_count == 0 || block_count == 0 || result_column_count == 0) {
        utility::LogError("Input tensors should have no zero dimensions.");
    }
    // A and B data will get manipulated in-place by LAPACK routines, make clones
	o3c::Tensor A_blocks_transposed = A_blocks.Clone();
    void *A_blocks_data = A_blocks_transposed.GetDataPtr();
	X = B.Clone();
    void *B_data = X.GetDataPtr();

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        solve_on_cuda(A_blocks_data, B_data, block_row_count, result_column_count,
                      block_count, data_type, device);
#else
        open3d::utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        solve_on_cpu(A_blocks_data, B_data, block_row_count, result_column_count, block_count,
                     data_type, device);
    }
}

} // namespace nnrt::core::linalg



