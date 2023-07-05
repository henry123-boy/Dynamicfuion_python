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
#include "MatmulBlockSparse.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::linalg {
std::tuple<open3d::core::Tensor, open3d::core::Tensor> MatmulBlockSparseRowWise(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates
) {

	std::tuple<open3d::core::Tensor, open3d::core::Tensor> matrices_and_coordinates;
	core::ExecuteOnDevice(
			blocks_b.GetDevice(),
			[&]() {
				matrices_and_coordinates = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CPU>(
						blocks_a, blocks_b, blocks_b_coordinates, true
				);
			},
			[&]() {
				NNRT_IF_CUDA(
						matrices_and_coordinates = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CUDA>(
								blocks_a, blocks_b, blocks_b_coordinates, true
						);
				);
			}
	);
	return matrices_and_coordinates;
}

open3d::core::Tensor MatmulBlockSparseRowWisePadded(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_coordinates
) {
	open3d::core::Tensor matrices, coordinates;
	core::ExecuteOnDevice(
			blocks_b.GetDevice(),
			[&]() {
				std::tie(matrices, coordinates) = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CPU>(
						blocks_a, blocks_b, blocks_b_coordinates, false
				);
			},
			[&]() {
				NNRT_IF_CUDA(
						std::tie(matrices, coordinates) = internal::MatmulBlockSparseRowWise<open3d::core::Device::DeviceType::CUDA>(
								blocks_a, blocks_b, blocks_b_coordinates, false
						);
				);
			}
	);
	return matrices;
}

}// namespace nnrt::core::linalg