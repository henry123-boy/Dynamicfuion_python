//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/13/22.
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
// stdlib includes

// third-party includes

// local includes
#include "core/functional/MaskingImpl.h"

namespace nnrt::core::functional {

template open3d::core::Tensor
SetMaskedToValue<float>(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, float element);

template open3d::core::Tensor
SetMaskedToValue<double>(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, double element);

template open3d::core::Tensor
SetMaskedToValue<int>(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, int element);

template open3d::core::Tensor
SetMaskedToValue<long int>(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, long int element);

template open3d::core::Tensor
SetMaskedToValue<long long int>(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, long long int element);

template open3d::core::Tensor
ReplaceValue<float>(open3d::core::Tensor& tensor, float old_value, float new_value);

template open3d::core::Tensor
ReplaceValue<double>(open3d::core::Tensor& tensor, double old_value, double new_value);

template open3d::core::Tensor
ReplaceValue<int>(open3d::core::Tensor& tensor, int old_value, int new_value);

template open3d::core::Tensor
ReplaceValue<long int>(open3d::core::Tensor& tensor, long int old_value, long int new_value);

template open3d::core::Tensor
ReplaceValue<long long int>(open3d::core::Tensor& tensor, long long int old_value, long long int new_value);

}// namespace nnrt::core::functional