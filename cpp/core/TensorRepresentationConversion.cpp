//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/19/22.
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
#include "core/TensorRepresentationConversion.h"

namespace nnrt::core {
template<>
open3d::core::Dtype GetDType<float>() {
    return open3d::core::Float32;
}

template<>
open3d::core::Dtype GetDType<int64_t>() {
    return open3d::core::Float64;
}
} // namespace nnrt::core
