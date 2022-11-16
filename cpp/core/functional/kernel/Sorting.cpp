//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/4/22.
//  Copyright (c) 2022 Gregory Kramida
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
#include "core/functional/kernel/Sorting.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::functional::kernel {

void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	ExecuteOnDevice(
			unsorted.GetDevice(),
			[&]() { SortTensorAlongLastDimension<open3d::core::Device::DeviceType::CPU>(sorted, unsorted); },
			[&]() { NNRT_IF_CUDA(SortTensorAlongLastDimension<open3d::core::Device::DeviceType::CUDA>(sorted, unsorted);); }
	);
}

} // namespace nnrt::core::functional::kernel