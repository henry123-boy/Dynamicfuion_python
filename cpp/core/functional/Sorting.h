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

namespace nnrt::core::functional {

//TODO: generalize these to "along any dimension/axis" -- will involve adaptation of some row structures/pointers to use in std::sort & BubbleSort,
// add corresponding arg, and remove "Dimension" from name
open3d::core::Tensor SortTensorAlongLastDimension(const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order = SortOrder::ASC);
std::tuple<open3d::core::Tensor, open3d::core::Tensor> SortTensorAlongLastDimensionByKey(const open3d::core::Tensor& values, const open3d::core::Tensor& keys, bool non_negative_first, SortOrder order = SortOrder::ASC);
open3d::core::Tensor ArgSortTensorAlongLastDimension(const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order = SortOrder::ASC);

open3d::core::Tensor SortTensorByColumn(const open3d::core::Tensor& unsorted, int column);
open3d::core::Tensor SortTensorByColumns(const open3d::core::Tensor& unsorted, const open3d::core::SizeVector& columns);

open3d::core::Tensor ArgSortByColumn(const open3d::core::Tensor& unsorted, int column);

} // nnrt::core::functional
