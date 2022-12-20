//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/20/22.
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
#include "core/GetDType.h"

namespace nnrt::core {

template<>
open3d::core::Dtype GetDType<double>() {
    return open3d::core::Float64;
}

template<>
open3d::core::Dtype GetDType<float>() {
    return open3d::core::Float32;
}

template<>
open3d::core::Dtype GetDType<int64_t>() {
    return open3d::core::Float64;
}

template<>
open3d::core::Dtype GetDType<int32_t>() {
    return open3d::core::Int32;
}

template<>
open3d::core::Dtype GetDType<int16_t>() {
    return open3d::core::Int16;
}

template<>
open3d::core::Dtype GetDType<int8_t>() {
    return open3d::core::Int8;
}

template<>
open3d::core::Dtype GetDType<uint64_t>() {
    return open3d::core::UInt64;
}

template<>
open3d::core::Dtype GetDType<uint32_t>() {
    return open3d::core::UInt32;
}

template<>
open3d::core::Dtype GetDType<uint16_t>() {
    return open3d::core::UInt16;
}

template<>
open3d::core::Dtype GetDType<uint8_t>() {
    return open3d::core::UInt8;
}

template<>
open3d::core::Dtype GetDType<bool>() {
    return open3d::core::Bool;
}



} // namespace nnrt::core