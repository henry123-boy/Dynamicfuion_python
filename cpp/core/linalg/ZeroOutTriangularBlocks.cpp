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
#include "ZeroOutTriangularBlocks.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace nnrt::core::linalg {

void ZeroOutTriangularBlocks(open3d::core::Tensor& blocks, nnrt::core::linalg::UpLoTriangular up_lo_triangular) {
	nnrt::core::ExecuteOnDevice(
			blocks.GetDevice(),
			[&] { internal::ZeroOutTriangularBlocks<o3c::Device::DeviceType::CPU>(blocks, up_lo_triangular); },
			[&] { NNRT_IF_CUDA(internal::ZeroOutTriangularBlocks<o3c::Device::DeviceType::CUDA>(blocks, up_lo_triangular);); }
	);
}

} // namespace nnrt::core::linalg
