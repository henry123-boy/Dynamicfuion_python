//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/14/21.
//  Copyright (c) 2021 Gregory Kramida
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

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
# define NNRT_DEVICE_WHEN_CUDACC __device__
# define NNRT_HOST_DEVICE_WHEN_CUDACC __host__ __device__
# define NNRT_HOST_WHEN_CUDACC __host__
# define NNRT_LAMBDA_CAPTURE_CLAUSE [=]
#else
# define NNRT_DEVICE_WHEN_CUDACC
# define NNRT_HOST_WHEN_CUDACC
# define NNRT_HOST_DEVICE_WHEN_CUDACC
# define NNRT_LAMBDA_CAPTURE_CLAUSE [&]
#endif