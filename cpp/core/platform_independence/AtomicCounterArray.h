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
#ifdef __CUDACC__
#include <cuda/std/cstdint>
#include <cuda_runtime.h>
#else

#include <vector>
#include <atomic>

#endif

// third-party includes
#include <open3d/core/Tensor.h>

// local includes

namespace nnrt::core {
#ifdef __CUDACC__
struct AtomicCounterArray {
//public:
    __host__
    AtomicCounterArray(uint32_t size_): size(size_) {
        cudaMalloc(&counters, sizeof(int) * size_);
        Reset();
    }
    __host__ __device__
    ~AtomicCounterArray() {
        cudaFree(counters);
    }
    __host__
    void Reset() {
        cudaMemset(counters, 0, sizeof(int) * size);
    }

    __device__
    int FetchAdd(int counter_index, int amount) {
        return atomicAdd(counters + counter_index, amount);
    }

    __device__
    int FetchSub(int counter_index, int amount) {
        return atomicSub(counters + counter_index, amount);
    }

    open3d::core::Tensor AsTensor(bool clone = false) {
        open3d::core::Tensor wrapper(counters, open3d::core::Int32, {size}, {}, open3d::core::Device("CUDA:0"));
        return clone ? wrapper.Clone() : wrapper;
    }

//private:
    uint32_t size;
    int* counters;
};
#else

class AtomicCounterArray {
public:
    AtomicCounterArray(std::size_t size) : counters(size) {
        Reset();
    }

    int FetchAdd(int counter_index, int amount) {
        return counters[counter_index].fetch_add(amount);
    }

    int FetchSub(int counter_index, int amount) {
        return counters[counter_index].fetch_sub(amount);
    }

    void Reset() {
        for (auto& counter: counters) {
            counter = 0;
        }
    }

    open3d::core::Tensor AsTensor(bool clone = false) {
        open3d::core::Tensor wrapper
                (counters.data(), open3d::core::Int32, {static_cast<long long>(counters.size())}, {},
                 open3d::core::Device("CPU:0"));
        return clone ? wrapper.Clone() : wrapper;
    }

private:
    std::vector<std::atomic_int> counters;

};

#endif
} // namespace nnrt::core