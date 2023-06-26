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
#pragma once
// 3rd party
#include <open3d/core/Tensor.h>
// local
#include "core/functional/SortOrder.h"

namespace nnrt::core::functional::kernel {

void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order);

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order);

void SortTensorAlongLastDimensionByKey(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys,
		bool non_negative_first, SortOrder order
);

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimensionByKey(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys,
		bool non_negative_first, SortOrder order
);

void SortTensorAlongFirstDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder sort_order);

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongFirstDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder sort_order);


void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place = false);

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place = false);

void ArgSortTensorByColumn(open3d::core::Tensor& index, const open3d::core::Tensor& unsorted, int column);

template<open3d::core::Device::DeviceType TDeviceType>
void ArgSortTensorByColumn(open3d::core::Tensor& index, const open3d::core::Tensor& unsorted, int column);

} // namespace nnrt::core::functional::kernel