//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/12/22.
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


#include <cmath>
#include <algorithm>




#ifdef __CUDACC__
namespace nnrt{

// region ====================== MAX ================================
template<typename T>
NNRT_DEVICE_WHEN_CUDACC
inline
T max_device(const T& a, const T& b){
	return a > b ? a : b;
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
float max_device<float>(const float& a, const float& b){
	return fmaxf(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
int max_device<int>(const int& a, const int& b){
	return max(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
long int max_device<long int>(const long int& a, const long int& b){
	return max(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
long long int max_device<long long int>(const long long int& a, const long long int& b){
	return max(a, b);
}

template<typename T>
__host__ __device__
inline
T max(const T& a, const T& b){
#ifdef  __CUDA_ARCH__
	return max_device<T>(a,b);
#else
	return std::max(a, b);
#endif
}
// endregion  =======================================================
// region ======================== MIN ==============================
template<typename T>
NNRT_DEVICE_WHEN_CUDACC
inline
T min_device(const T& a, const T& b){
	return a < b ? a : b;
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
float min_device<float>(const float& a, const float& b){
	return fminf(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
int min_device<int>(const int& a, const int& b){
	return min(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
long int min_device<long int>(const long int& a, const long int& b){
	return min(a, b);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline
long long int min_device<long long int>(const long long int& a, const long long int& b){
	return min(a, b);
}

template<typename T>
__host__ __device__
inline
T min(const T& a, const T& b){
#ifdef  __CUDA_ARCH__
	return min_device<T>(a,b);
#else
	return std::min(a, b);
#endif
}

// endregion  =======================================================
}// namespace nnrt
#else

namespace nnrt {

template<typename T>
const T& max(const T& a, const T& b){
	return std::max<T>(a, b);
}

template<typename T>
const T& min(const T& a, const T& b){
	return std::min<T>(a, b);
}

} // namespace nnrt
#endif