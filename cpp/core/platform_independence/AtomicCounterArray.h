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

namespace nnrt::core {

template<open3d::core::Device::DeviceType TDeviceType>
class AtomicCounterArray;

#ifdef __CUDACC__
template<>
class AtomicCounterArray<open3d::core::Device::DeviceType::CUDA> {
public:
    AtomicCounterArray(uint32_t size):
        counters(open3d::core::Tensor::Zeros({static_cast<int64_t>(size)}, open3d::core::Int32, open3d::core::Device("CUDA:0"))),
        counters_data(counters.GetDataPtr<int32_t>()){
    }

    void Reset(){
        counters.Fill(0);
    }

    __device__
    int GetCount(int counter_index) const {
        return counters_data[counter_index];
    }

    //FIXME note: making this (and next method) const is kind of a hack, since the function does modify the object's data,
    // but this is the only way to get around Open3D's ParallelFor not being able to accept mutable lambdas right now.
    __device__
    int FetchAdd(int counter_index, int amount) const {
        return atomicAdd(counters_data + counter_index, amount);
    }

    __device__
    int FetchSub(int counter_index, int amount) const {
        return atomicSub(counters_data + counter_index, amount);
    }

    open3d::core::Tensor AsTensor(bool clone = false) {
        return clone ? counters.Clone() : counters;
    }

private:
    open3d::core::Tensor counters;
    int* counters_data;
};
#else
template<>
class AtomicCounterArray<open3d::core::Device::DeviceType::CPU> {
public:
    AtomicCounterArray(std::size_t size) : counters(size) {
        Reset();
    }

    int FetchAdd(int counter_index, int amount) {
        return counters[counter_index].fetch_add(amount);
    }

    int GetCount(int counter_index) const {
        return counters[counter_index].load();
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