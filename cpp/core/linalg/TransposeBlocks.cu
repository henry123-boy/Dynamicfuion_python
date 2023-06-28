//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/28/23.
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
#include <open3d/core/Dispatch.h>
#include <open3d/core/CUDAUtils.h>

// local includes
#include "core/linalg/TransposeBlocks.h"
#include "core/linalg/LinalgKernels.cuh"
#include "core/linalg/PointerAggregationForBatchOperationsCUDA.cuh"


namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg {


template<typename TElement>
void TransposeBlocksInPlace_TypeDispatched(open3d::core::Tensor& blocks) {
	auto device = blocks.GetDevice();

	if (device.IsCPU()) {
		utility::LogError("TransposeBlocksInPlace not implemented for CPU. Use regular Open3D routine, e.g. `blocks = blocks.Transpose(1, 2)`");
	}
	int64_t block_size = blocks.GetShape(1);
	int64_t block_count = blocks.GetShape(0);
	o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });

	auto block_data = blocks.GetDataPtr<TElement>();

	TElement** block_array_device;
	auto size_of_pointer_array = block_count * sizeof(TElement*);
	OPEN3D_CUDA_CHECK(cudaMalloc(&block_array_device, size_of_pointer_array));

	internal::GetMatrixPointersFromContiguousArrayOfMatrices_CUDA(block_array_device, block_data, block_size, block_size, block_count, device);
	internal::transpose_batched_cuda<TElement>(block_array_device, block_count, block_size, device);

	OPEN3D_CUDA_CHECK(cudaFree(block_array_device));
}

void TransposeBlocksInPlace(open3d::core::Tensor& blocks) {
	DISPATCH_DTYPE_TO_TEMPLATE(
			blocks.GetDtype(),
			[&] {
				TransposeBlocksInPlace_TypeDispatched<scalar_t>(blocks);
			}
	);
}

} // namespace nnrt::core::linalg