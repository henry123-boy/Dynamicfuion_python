//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/11/23.
//  Copyright (c) 2023 Gregory Kramida
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
#ifdef __CUDACC__
#ifndef BUILD_CUDA_MODULE
#define BUILD_CUDA_MODULE
#endif
#include <open3d/core/CUDAUtils.h>
#endif
#include <open3d/utility/Parallel.h>
#include <open3d/utility/Logging.h>
#include <open3d/core/ParallelFor.h>

// local includes


namespace nnrt::core{
#ifdef __CUDACC__

static constexpr int64_t NNRT_PARFOR_BLOCK = 128;
static constexpr int64_t NNRT_PARFOR_THREAD = 4;

/// Run a function in parallel on CUDA.
template <typename TFunction>
void ParallelForMutableCUDA_(const open3d::core::Device& device, int64_t work_unit_count, TFunction&& func) {
    if (device.GetType() != open3d::core::Device::DeviceType::CUDA) {
        open3d::utility::LogError("ParallelFor for CUDA cannot run on device {}.",
                          device.ToString());
    }
    if (work_unit_count == 0) {
        return;
    }

    open3d::core::CUDAScopedDevice scoped_device(device);
    int64_t items_per_block = NNRT_PARFOR_BLOCK * NNRT_PARFOR_THREAD;
    int64_t grid_size = (work_unit_count + items_per_block - 1) / items_per_block;

    open3d::core::ElementWiseKernel_<NNRT_PARFOR_BLOCK, NNRT_PARFOR_THREAD>
            <<<grid_size, NNRT_PARFOR_BLOCK, 0, open3d::core::cuda::GetStream()>>>(
                    work_unit_count, func);
    open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("ParallelFor failed.");
}

#else

/// Run a function in parallel on CPU.
template <typename TFunction>
void ParallelForMutableCPU_(const open3d::core::Device& device, int64_t work_unit_count, TFunction&& func) {
    if (!device.IsCPU()) {
        open3d::utility::LogError("ParallelForMutable for CPU cannot run on device {}.",
                          device.ToString());
    }
    if (work_unit_count == 0) {
        return;
    }

#pragma omp parallel for num_threads(open3d::utility::EstimateMaxThreads())
    for (int64_t i = 0; i < work_unit_count; ++i) {
        func(i);
    }
}



#endif

template <typename TFunction>
void ParallelForMutable(const open3d::core::Device& device, int64_t work_unit_count, TFunction&& function) {
#ifdef __CUDACC__
    ParallelForMutableCUDA_(device, work_unit_count, function);
#else
    ParallelForMutableCPU_(device, work_unit_count, function);
#endif
}

} // namespace nnrt::core