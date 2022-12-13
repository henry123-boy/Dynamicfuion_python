//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
// local includes

namespace nnrt::core::functional{
template<typename TElement>
open3d::core::Tensor  SetMaskedToValue(open3d::core::Tensor& tensor, const open3d::core::Tensor& mask, TElement element);

template<typename TElement>
open3d::core::Tensor ReplaceValue(open3d::core::Tensor& tensor, TElement old_value, TElement new_value);

} // namespace nnrt::core::functional

