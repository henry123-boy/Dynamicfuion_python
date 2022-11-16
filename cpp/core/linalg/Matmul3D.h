//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/12/22.
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

#include <open3d/core/Tensor.h>

namespace nnrt::core::linalg {

void Matmul3D(open3d::core::Tensor& output, const open3d::core::Tensor& array_of_matrices_A, const open3d::core::Tensor& array_of_matrices_B);

template<open3d::core::Device::DeviceType DeviceType>
void Matmul3D(const void* A, const void* B, void* C, int64_t m, int64_t k, int64_t n,
              int64_t batch_size, open3d::core::Dtype dtype);

} // nnrt::core::linalg