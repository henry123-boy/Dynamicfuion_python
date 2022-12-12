//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/6/22.
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
#ifdef __CUDACC__
#include <cuda/std/array>
namespace nnrt {
template<typename T, cuda::std::size_t N>
using array = cuda::std::array<T, N>;
} // namespace nnrt
#else
#include <array>
namespace nnrt {
template<typename T, std::size_t N>
using array = std::array<T, N>;
} // namespace nnrt
#endif