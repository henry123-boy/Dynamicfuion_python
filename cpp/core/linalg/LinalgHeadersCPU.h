//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/13/22.
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

#pragma once

#ifdef USE_BLAS
#define NNRT_CPU_LINALG_INT int32_t
#define lapack_int int32_t
#include <cblas.h>
#include <lapacke.h>
#else
#include <mkl.h>
static_assert(
        sizeof(MKL_INT) == 8,
        "MKL_INT must be 8 bytes: please link with MKL 64-bit int library.");
#define NNRT_CPU_LINALG_INT MKL_INT
#endif