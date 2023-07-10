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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include "UpLoTriangular.h"

// local includes

namespace nnrt::core::linalg {

open3d::core::Tensor InvertPositiveSemidefiniteBlocks(const open3d::core::Tensor& blocks);
open3d::core::Tensor InvertTriangularBlocks(const open3d::core::Tensor& blocks, nnrt::core::linalg::UpLoTriangular uplo);

namespace internal {

void SolveBlocksCPU(
		void* a_block_data,
		void* b_block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
);

void SolveBlocksCUDA(
		void* a_block_data,
		void* b_block_data,
		int64_t block_size,
		int64_t block_count,
		open3d::core::Dtype data_type,
		const open3d::core::Device& device
);

void InvertTriangularBlocksCPU(
		void* A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Dtype& data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
);

void InvertTriangularBlocksCUDA(
		void* A_block_data,
		int64_t block_size,
		int64_t block_count,
		const open3d::core::Dtype& data_type,
		const open3d::core::Device& device,
		nnrt::core::linalg::UpLoTriangular uplo
);

} // internal
} // nnrt::core::linalg