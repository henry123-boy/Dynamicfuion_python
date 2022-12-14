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

namespace nnrt{
#ifdef __CUDACC__


__device__
inline float saturatef(float x){
    return __saturatef(x);
}

__device__
inline float fmin3f(const float& a, const float& b, const float& c){
    return fminf(fminf(a,b),c);
}

__device__
inline float fmax3f(const float& a, const float& b, const float& c){
    return fmaxf(fmaxf(a,b),c);
}

#else

inline float saturatef(float x){
    return fminf(fmaxf(x, 0.00f), 1.00f);
}

inline const float& fmin3f(const float& a, const float& b, const float& c){
    return std::min(std::min(a,b),c);
}

inline const float& fmax3f(const float& a, const float& b, const float& c){
    return std::max(std::max(a,b),c);
}


#endif
}// namespace nnrt