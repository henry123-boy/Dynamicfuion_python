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
// stdlib includes

// third-party includes

// local includes
#include "InvertBlocks.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg {

open3d::core::Tensor InvertTriangularBlocks(
		const open3d::core::Tensor& blocks,
		nnrt::core::linalg::UpLoTriangular uplo
) {
	// TODO: to avoid DRY violation / code overlap w/ FactorizeBlocksCholesky and SolveBlockDiagonalGeneric,
	//  block operations like this can be outsourced to another, even more generic function (or multiple other functions).
	o3c::AssertTensorDtypes(blocks, { o3c::Float32, o3c::Float64 });
	const o3c::Device device = blocks.GetDevice();
	const o3c::Dtype data_type = blocks.GetDtype();


	o3c::SizeVector blocks_shape = blocks.GetShape();
	if (blocks_shape.size() != 3) {
		utility::LogError("Tensor blocks must have three dimensions, got {}.", blocks_shape.size());
	}
	if (blocks_shape[1] != blocks_shape[2]) {
		utility::LogError("Tensor blocks must consist of square blocks, "
		                  "i.e. dimensions at indices 1 and 2 must be equal. "
		                  "Got dimensions: {} and {}..", blocks_shape[1], blocks_shape[2]);
	}

	int64_t block_size = blocks_shape[1];
	int64_t block_count = blocks_shape[0];

	if (block_size == 0 || block_count == 0) {
		utility::LogError("Input tensor should have no zero dimensions.");
	}
	// data will get manipulated in-place by LAPACK routines, make clone
	o3c::Tensor inverted_blocks = blocks.Clone().Transpose(1, 2);
	void* block_data = inverted_blocks.GetDataPtr();

	if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
		//TODO: revert InvertTriangularBlocksCUDA signature back to just use one array (and modify in-place). Doesn't seem to make any difference.
		internal::InvertTriangularBlocksCUDA(blocks.GetDataPtr(), block_data, block_size, block_count, data_type, device, uplo);
#else
		open3d::utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		internal::InvertTriangularBlocksCPU(block_data, block_size, block_count, data_type, device, uplo);
	}

	// Perform column- to row-major reordering using axis swap
	inverted_blocks = inverted_blocks.Transpose(1, 2);
	return inverted_blocks;
}

} // namespace nnrt::core::linalg

