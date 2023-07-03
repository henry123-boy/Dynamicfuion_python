//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/21/23.
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
#include "DiagonalBlocks.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::core::linalg{

void FillInDiagonalBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks) {
	nnrt::core::ExecuteOnDevice(
			blocks.GetDevice(),
			[&] { internal::FillInDiagonalBlocks<o3c::Device::DeviceType::CPU>(matrix, blocks); },
			[&] { NNRT_IF_CUDA(internal::FillInDiagonalBlocks<o3c::Device::DeviceType::CUDA>(matrix, blocks);); }
	);
}

open3d::core::Tensor GetDiagonalBlocks(const open3d::core::Tensor& matrix, int block_size) {
	o3c::Tensor blocks;
	nnrt::core::ExecuteOnDevice(
			matrix.GetDevice(),
			[&] { blocks = internal::GetDiagonalBlocks<o3c::Device::DeviceType::CPU>(matrix, block_size); },
			[&] { NNRT_IF_CUDA( blocks = internal::GetDiagonalBlocks<o3c::Device::DeviceType::CUDA>(matrix, block_size);); }
	);
	return blocks;
}

} // namespace nnrt::core::linalg