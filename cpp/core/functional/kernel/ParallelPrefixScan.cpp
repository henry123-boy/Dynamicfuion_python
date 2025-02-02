//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/26/23.
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
#include "core/DeviceSelection.h"
#include "core/functional/kernel/ParallelPrefixScan.h"

namespace o3c = open3d::core;

namespace nnrt::core::functional::kernel {

void ExclusiveParallelPrefixSum1D(open3d::core::Tensor& prefix_sum, const open3d::core::Tensor& source) {
	core::ExecuteOnDevice(
			source.GetDevice(),
			[&] {
				ExclusiveParallelPrefixSum1D<o3c::Device::DeviceType::CPU>(prefix_sum, source);
			},
			[&] {
				NNRT_IF_CUDA(
						ExclusiveParallelPrefixSum1D<o3c::Device::DeviceType::CUDA>(prefix_sum, source);
				);
			}

	);
}

void InclusiveParallelPrefixSum1D(open3d::core::Tensor& prefix_sum, const open3d::core::Tensor& source) {
	core::ExecuteOnDevice(
			source.GetDevice(),
			[&] {
				InclusiveParallelPrefixSum1D<o3c::Device::DeviceType::CPU>(prefix_sum, source);
			},
			[&] {
				NNRT_IF_CUDA(
						InclusiveParallelPrefixSum1D<o3c::Device::DeviceType::CUDA>(prefix_sum, source);
				);
			}

	);
}

} // namespace nnrt::core::functional::kernel
