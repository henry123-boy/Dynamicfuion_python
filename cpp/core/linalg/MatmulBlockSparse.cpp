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

std::tuple<open3d::core::Tensor, open3d::core::Tensor> MatmulBlockSparse(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_a_breadboard,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& blocks_b,
		const open3d::core::Tensor& blocks_b_breadboard,
		MatrixPreprocessingOperation matrix_b_preprocessing
) {
	std::tuple<open3d::core::Tensor, open3d::core::Tensor> blocks_and_coordinates;
	core::ExecuteOnDevice(
			blocks_b.GetDevice(),
			[&]() {
				blocks_and_coordinates = internal::MatmulBlockSparse<open3d::core::Device::DeviceType::CPU>(
						blocks_a, blocks_a_breadboard, matrix_a_preprocessing, blocks_b,
						blocks_b_breadboard, matrix_b_preprocessing
				);
			},
			[&]() {
				NNRT_IF_CUDA(
						blocks_and_coordinates = internal::MatmulBlockSparse<open3d::core::Device::DeviceType::CUDA>(
								blocks_a, blocks_a_breadboard, matrix_a_preprocessing, blocks_b,
								blocks_b_breadboard, matrix_b_preprocessing
						);
				);
			}
	);
	return blocks_and_coordinates;
}

open3d::core::Tensor BlockSparseAndVectorProduct(
		const open3d::core::Tensor& blocks_a,
		const open3d::core::Tensor& blocks_a_breadboard,
		MatrixPreprocessingOperation matrix_a_preprocessing,
		const open3d::core::Tensor& vector_b
) {
	open3d::core::Tensor out_vector;
	core::ExecuteOnDevice(
			blocks_a.GetDevice(),
			[&]() {
				internal::BlockSparseAndVectorProduct<open3d::core::Device::DeviceType::CPU>(
						out_vector,
						blocks_a, blocks_a_breadboard, matrix_a_preprocessing, vector_b
				);
			},
			[&]() {
				NNRT_IF_CUDA(
						out_vector = internal::BlockSparseAndVectorProduct<open3d::core::Device::DeviceType::CUDA>(
								out_vector,
								blocks_a, blocks_a_breadboard, matrix_a_preprocessing, vector_b
						);
				);
			}
	);
	return out_vector;
}


}// namespace nnrt::core::linalg
