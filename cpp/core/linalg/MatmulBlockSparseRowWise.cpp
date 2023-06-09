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
// third-party includes

// local includes
#include "MatmulBlockSparseRowWise.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::linalg {
open3d::core::Tensor MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates
) {

	open3d::core::Tensor matrices;
	core::ExecuteOnDevice(
			blocks_b.GetDevice(),
			[&]() {
				matrices = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CPU>(blocks_a,
				                                                                          blocks_b, blocks_b_coordinates);
			},
			[&]() {
				NNRT_IF_CUDA(
						matrices = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CUDA>(blocks_a,
						                                                                           blocks_b, blocks_b_coordinates);
				);
			}
	);
	return matrices;
}

}// namespace nnrt::core::linalg
