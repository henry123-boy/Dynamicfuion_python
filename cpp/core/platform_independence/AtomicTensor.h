//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/30/23.
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
// stdlib & CUDA includes
#ifdef __CUDACC__
#include <cuda/std/cstdint>
#include <cuda_runtime.h>
#else
#include <vector>
#include <atomic>
#endif

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/utility/Logging.h>

// local includes
#include "core/GetDType.h"
#include "core/platform_independence/Atomics.h"

namespace nnrt::core {

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
class AtomicTensor;
#ifdef __CUDACC__
template<typename TElement>
class AtomicTensor<open3d::core::Device::DeviceType::CUDA, TElement> {
	static_assert(std::is_fundamental<TElement>::value && (std::is_floating_point<TElement>::value || std::is_integral<TElement>::value));
public:
	const open3d::core::Device::DeviceType DeviceType = open3d::core::Device::DeviceType::CUDA;
	AtomicTensor(const open3d::core::SizeVector& shape, const open3d::core::Device& device):
			atomic_storage(open3d::core::Tensor::Zeros(shape, GetDType<TElement>(), device)),
			atomic_data(this->atomic_storage.template GetDataPtr<TElement>()){
		if(device.GetType() != DeviceType){
			open3d::utility::LogError("Device, {}, needs to have matching device type with {}.", device.GetType(), DeviceType);
		}
	}

	void Reset(){
		atomic_storage.Fill(0);
	}

	__device__
	inline TElement GetValue(int index) const {
		return atomic_data[index];
	}

	//FIXME note: making this (and next method) const is kind of a hack, since the function does modify the object's data,
	// but this is the only way to get around Open3D's ParallelFor not being able to accept mutable lambdas right now.
	__device__
	inline TElement FetchAdd(int index, TElement amount) const {
		return atomicAdd(atomic_data + index, amount);
	}

	__device__
	inline TElement FetchSub(int index, TElement amount) const {
		return atomicAdd(atomic_data + index, -amount);
	}

	TElement* GetDataPtr() {
		return atomic_storage.GetDataPtr<TElement>();
	}

	open3d::core::Tensor AsTensor(bool clone = false) {
		return clone ? atomic_storage.Clone() : atomic_storage;
	}

private:
	open3d::core::Tensor atomic_storage;
	TElement* atomic_data;
};

#else

template<typename TElement>
class AtomicTensor<open3d::core::Device::DeviceType::CPU, TElement>{
	static_assert(std::is_fundamental<TElement>::value && (std::is_floating_point<TElement>::value || std::is_integral<TElement>::value));
public:
	const open3d::core::Device::DeviceType DeviceType = open3d::core::Device::DeviceType::CPU;
	AtomicTensor(const open3d::core::SizeVector& shape, const open3d::core::Device& device ):
			atomic_storage(shape.NumElements()), shape(shape), device(device){
		if(device.GetType() != DeviceType){
			open3d::utility::LogError("Device, {}, needs to have matching device type with {}.", device.GetType(), DeviceType);
		}
		Reset();
	};
	inline TElement FetchAdd(int index, TElement amount) {
		return atomicAdd_CPU(atomic_storage[index], amount);
	}

	inline TElement GetValue(int index) const {
		return atomic_storage[index].load();
	}

	inline TElement FetchSub(int index, TElement amount) {
		return atomicSub_CPU(atomic_storage[index], amount);
	}

	void Reset() {
		for (auto& value: atomic_storage) {
			value = 0;
		}
	}

	TElement* GetDataPtr() {
		return reinterpret_cast<float*>(atomic_storage.data());
	}

	open3d::core::Tensor AsTensor(bool clone = false) {
		open3d::core::Tensor wrapper
				(atomic_storage.data(), GetDType<TElement>(), shape, {}, this->device);
		return clone ? wrapper.Clone() : wrapper;
	}

private:
	std::vector<std::atomic<TElement>> atomic_storage;
	open3d::core::SizeVector shape;
	open3d::core::Device device;
};

#endif


} //namespace nnrt::core