// Copyright (c) 2023. Gregory Kramida

//  ================================================================
// Created by Gregory Kramida (https://github.com/Algomorph) on 4/4/23.
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
// local includes
#ifdef BUILD_CUDA_MODULE
#define NNRT_IF_CUDA(...) __VA_ARGS__ static_assert(true)
#else
#define NNRT_IF_CUDA(...) static_assert(true)
#endif