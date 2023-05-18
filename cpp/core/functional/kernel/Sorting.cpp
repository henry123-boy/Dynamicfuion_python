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

void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool positive_first, SortOrder order) {
	ExecuteOnDevice(
			unsorted.GetDevice(),
			[&]() { SortTensorAlongLastDimension<open3d::core::Device::DeviceType::CPU>(sorted, unsorted, positive_first, order); },
			[&]() { NNRT_IF_CUDA(SortTensorAlongLastDimension<open3d::core::Device::DeviceType::CUDA>(sorted, unsorted, positive_first, order);); }
	);
}

void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place) {
	ExecuteOnDevice(
			unsorted.GetDevice(),
			[&]() { SortTensorByColumn<open3d::core::Device::DeviceType::CPU>(sorted, unsorted, column, in_place); },
			[&]() { NNRT_IF_CUDA(SortTensorByColumn<open3d::core::Device::DeviceType::CUDA>(sorted, unsorted, column, in_place);); }
	);
}

void ArgSortTensorByColumn(open3d::core::Tensor& index, const open3d::core::Tensor& unsorted, int column) {
	ExecuteOnDevice(
			unsorted.GetDevice(),
			[&]() { ArgSortTensorByColumn<open3d::core::Device::DeviceType::CPU>(index, unsorted, column); },
			[&]() { NNRT_IF_CUDA(ArgSortTensorByColumn<open3d::core::Device::DeviceType::CUDA>(index, unsorted, column);); }
	);
}

} // namespace nnrt::core::functional::kernel