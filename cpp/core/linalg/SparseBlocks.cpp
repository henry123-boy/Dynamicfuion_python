//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/3/23.
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
// local includes
#include "core/linalg/SparseBlocks.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::core::linalg {


void FillInSparseBlocks(open3d::core::Tensor& matrix, const open3d::core::Tensor& blocks, const open3d::core::Tensor& coordinates, bool transpose) {
	nnrt::core::ExecuteOnDevice(
			blocks.GetDevice(),
			[&] { internal::FillInSparseBlocks<o3c::Device::DeviceType::CPU>(matrix, blocks, coordinates, transpose); },
			[&] { NNRT_IF_CUDA(internal::FillInSparseBlocks<o3c::Device::DeviceType::CUDA>(matrix, blocks, coordinates, transpose);); }
	);
}

open3d::core::Tensor GetSparseBlocks(const open3d::core::Tensor& matrix, int block_size, const open3d::core::Tensor& coordinates) {
	o3c::Tensor blocks;
	nnrt::core::ExecuteOnDevice(
			matrix.GetDevice(),
			[&] { blocks = internal::GetSparseBlocks<o3c::Device::DeviceType::CPU>(matrix, block_size, coordinates); },
			[&] { NNRT_IF_CUDA( blocks = internal::GetSparseBlocks<o3c::Device::DeviceType::CUDA>(matrix, block_size, coordinates);); }
	);
	return blocks;
}

} // namespace nnrt::core::linalg
