//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/25/23.
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

// third-party includes
#include <open3d/core/Tensor.h>

namespace nnrt::core {

typedef union {
	struct {
		float key;
		int index;
	} pair;
	unsigned long long int bitstring;
} AtomicKeyValue;

template<open3d::core::Device::DeviceType TDeviceType>
class AtomicKeyIndexArray;


#ifdef __CUDACC__
__device__ unsigned long long int atomicMin(unsigned long long int* address, float key, int index) {
	AtomicKeyValue expected, discovered;
	expected.pair.key = key;
	expected.pair.index = index;
	discovered.bitstring = *address;
	while (discovered.pair.key > key)
		discovered.bitstring = atomicCAS(address, discovered.bitstring, expected.bitstring);
	return discovered.bitstring;
}

__device__ unsigned long long int atomicMax(unsigned long long int* address, float key, int index) {
	AtomicKeyValue expected, discovered;
	expected.pair.key = key;
	expected.pair.index = index;
	discovered.bitstring = *address;
	while (discovered.pair.key < key)
		discovered.bitstring = atomicCAS(address, discovered.bitstring, expected.bitstring);
	return discovered.bitstring;
}

template<>
class AtomicKeyIndexArray<open3d::core::Device::DeviceType::CUDA> {
public:
	AtomicKeyIndexArray(uint32_t size, const open3d::core::Device& device, bool min = true) :
			size(size),
			blob(std::make_shared<open3d::core::Blob>(size * sizeof(unsigned long long int), device)),
			data(reinterpret_cast<unsigned long long int*>(blob->GetDataPtr())),
			min(min) {
		Reset();
	}

	void Reset() {
		float initial_key = this->min ? INFINITY : -INFINITY;
		unsigned long long int* data_local = this->data;
		o3c::ParallelFor(
				blob->GetDevice(),
				size,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					AtomicKeyValue local;
					local.pair.index = -1;
					local.pair.key = initial_key;
					data_local[workload_idx] = local.bitstring;
				}
		);
	}

	__device__
	AtomicKeyValue Get(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at];
		return local;
	}

	__device__
	float GetKey(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at];
		return local.pair.key;
	}

	__device__
	int GetIndex(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at];
		return local.pair.index;
	}

	//FIXME note: making this (and next method) const is kind of a hack, since the function does modify the object's data,
	// but this is the only way to get around Open3D's ParallelFor not being able to accept mutable lambdas right now.
	__device__
	AtomicKeyValue FetchMin(int at, float key, int index) const {
		AtomicKeyValue local;
		local.bitstring = atomicMin(data + at, key, index);
		return local;
	}

	__device__
	AtomicKeyValue FetchMax(int at, float key, int index) const {
		AtomicKeyValue local;
		local.bitstring = atomicMax(data + at, key, index);
		return local;
	}

	__host__
	open3d::core::Tensor KeyTensor() {
		o3c::Tensor keys({size}, o3c::Float32, blob->GetDevice());
		auto key_data = keys.GetDataPtr<float>();
		unsigned long long int* data_local = this->data;
		o3c::ParallelFor(
				blob->GetDevice(),
				size,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					AtomicKeyValue local;
					local.bitstring = data_local[workload_idx];
					key_data[workload_idx] = local.pair.key;
				}
		);
		return keys;
	}

	__host__
	open3d::core::Tensor IndexTensor() {
		o3c::Tensor indices({size}, o3c::Int32, blob->GetDevice());
		auto index_data = indices.GetDataPtr<int32_t>();
		unsigned long long int* data_local = this->data;
		o3c::ParallelFor(
				blob->GetDevice(),
				size,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					AtomicKeyValue local;
					local.bitstring = data_local[workload_idx];
					index_data[workload_idx] = local.pair.index;
				}
		);
		return indices;
	}

private:
	uint32_t size;
	std::shared_ptr<open3d::core::Blob> blob;
	unsigned long long int* data;
	bool min;
};

#else

unsigned long long int atomicMin(std::atomic<unsigned long long int>& variable, float key, int index) {
	AtomicKeyValue expected, discovered;
	expected.pair.key = key;
	expected.pair.index = index;
	discovered.bitstring = variable.load();
	while (discovered.pair.key > key &&
		   !variable.compare_exchange_weak(discovered.bitstring, expected.bitstring, std::memory_order_relaxed, std::memory_order_relaxed));
	return discovered.bitstring;
}

unsigned long long int atomicMax(std::atomic<unsigned long long int>& variable, float key, int index) {
	AtomicKeyValue expected, discovered;
	expected.pair.key = key;
	expected.pair.index = index;
	discovered.bitstring = variable.load();
	while (discovered.pair.key < key &&
		   !variable.compare_exchange_weak(discovered.bitstring, expected.bitstring, std::memory_order_relaxed, std::memory_order_relaxed));
	return discovered.bitstring;
}

template<>
class AtomicKeyIndexArray<open3d::core::Device::DeviceType::CPU> {
public:
	AtomicKeyIndexArray(uint32_t size, const open3d::core::Device& device, bool min = true) :
			data(size), device(device), min(min) {
		Reset();
	}

	void Reset() {
		float initial_key = this->min ? INFINITY : -INFINITY;
		for (auto& item: data) {
			AtomicKeyValue local;
			local.pair.index = -1;
			local.pair.key = initial_key;
			item.store(local.bitstring);
		}
	}


	AtomicKeyValue Get(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at].load();
		return local;
	}

	float GetKey(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at].load();
		return local.pair.key;
	}

	int GetIndex(int at) const {
		AtomicKeyValue local;
		local.bitstring = data[at].load();
		return local.pair.index;
	}

	//FIXME note: making this (and next method) const is kind of a hack, since the function does modify the object's data,
	// but this is the only way to get around Open3D's ParallelFor not being able to accept mutable lambdas right now.
	AtomicKeyValue FetchMin(int at, float key, int index) {
		AtomicKeyValue local;
		local.bitstring = atomicMin(data[at], key, index);
		return local;
	}

	AtomicKeyValue FetchMax(int at, float key, int index) {
		AtomicKeyValue local;
		local.bitstring = atomicMax(data[at], key, index);
		return local;
	}

	open3d::core::Tensor KeyTensor() {
		o3c::Tensor keys({static_cast<int64_t>(data.size())}, o3c::Float32, device);
		auto key_data = keys.GetDataPtr<float>();
		o3c::ParallelFor(
				device,
				keys.GetLength(),
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					key_data[workload_idx] = GetKey(static_cast<int32_t>(workload_idx));
				}
		);
		return keys;
	}

	open3d::core::Tensor IndexTensor() {
		o3c::Tensor indices({static_cast<int64_t>(data.size())}, o3c::Int32, device);
		auto index_data = indices.GetDataPtr<int32_t>();
		o3c::ParallelFor(
				device,
				indices.GetLength(),
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					index_data[workload_idx] = GetIndex(static_cast<int32_t>(workload_idx));
				}
		);
		return indices;
	}

private:
	std::vector<std::atomic<unsigned long long int>> data;
	open3d::core::Device device;
	bool min;
};


#endif
} // namespace nnrt::core
