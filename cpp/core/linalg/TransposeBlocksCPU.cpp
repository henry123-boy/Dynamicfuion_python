//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/11/23.
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
// third-party includes
#include <open3d/utility/Parallel.h>

// local includes
#include "core/linalg/TransposeBlocks.h"
namespace o3c = open3d::core;
namespace utility = open3d::utility;


namespace nnrt::core::linalg::internal {
template<typename TElement>
inline void TransposeBlocksInPlaceCPU_TypeDispatched(open3d::core::Tensor& blocks){
	//TODO

/*	auto device = blocks.GetDevice();
	int64_t block_size = blocks.GetShape(1);
	int64_t block_count = blocks.GetShape(0);
	int64_t block_stride = block_size * block_size;
	o3c::AssertTensorShape(blocks, { block_count, block_size, block_size });

	auto block_data = blocks.GetDataPtr<TElement>();*/

	utility::LogError("Not implemented.");

/*#pragma omp parallel for schedule(static) num_threads(open3d::utility::EstimateMaxThreads()) \
    default(none) \
    firstprivate(block_count, block_size, block_stride) \
    shared(block_data)
	for (int i_matrix = 0; i_matrix < block_count; i_matrix++) {

	}*/
}

void TransposeBlocksInPlaceCPU(open3d::core::Tensor& blocks){
	DISPATCH_DTYPE_TO_TEMPLATE(
			blocks.GetDtype(),
			[&] {
				TransposeBlocksInPlaceCPU_TypeDispatched<scalar_t>(blocks);
			}
	);
}

} // namespace nnrt::core::linalg::internal