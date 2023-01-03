//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/17/22.
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

// ============== Kernel code usage ONLY ==============

#ifdef __CUDACC__
#include <cuda/std/tuple>
namespace nnrt {
template<typename... Ts>
using tuple = cuda::std::tuple<Ts...>;

template<typename... Ts>
__device__
inline cuda::std::tuple<Ts...> make_tuple(Ts... args){
	return cuda::std::make_tuple(args...);
}

template<int IItem, typename TTuple>
__device__
inline auto get(TTuple& tuple) {
	return cuda::std::get<IItem>(tuple);
}

} // namespace nnrt
#else
#include <tuple>
namespace nnrt {
template<typename... Ts>
using tuple = std::tuple<Ts...>;

template<typename... Ts>
inline std::tuple<Ts...> make_tuple(Ts... args){
	return std::make_tuple(args...);
}


template<int IItem, typename TTuple>
inline auto get(TTuple& tuple) {
	return std::get<IItem>(tuple);
}

} // namespace nnrt
#endif


