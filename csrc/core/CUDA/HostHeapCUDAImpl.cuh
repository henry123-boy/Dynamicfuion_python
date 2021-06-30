//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/29/21.
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

#include "core/Heap.h"
#include "core/CUDA/DeviceHeapCUDA.cuh"
#include <open3d/core/kernel/CUDALauncher.cuh>
#include <open3d/core/CUDAUtils.h>


namespace nnrt {
namespace core {

template<typename TKey, typename TValue>
__global__
void MakeMinHeap(IDeviceHeap*& device_heap, void* storage, const int capacity) {
	typedef core::KeyValuePair<TKey, TValue> DistanceIndexPair;
	typedef decltype(core::MinHeapKeyCompare<TKey, TValue>) Compare;

	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, DistanceIndexPair, Compare> HeapType;
	device_heap = reinterpret_cast<IDeviceHeap*>(malloc(sizeof(HeapType)));
	(*device_heap) = HeapType(
			reinterpret_cast<KeyValuePair<TKey, TValue>*>(storage),
			capacity,
			core::MinHeapKeyCompare<TKey, TValue>
	);
}

template<typename TKey, typename TValue>
__global__
void MakeMaxHeap(IDeviceHeap*& device_heap, void* storage, const int capacity) {
	typedef core::KeyValuePair<TKey, TValue> DistanceIndexPair;
	typedef decltype(core::MaxHeapKeyCompare<TKey, TValue>) Compare;

	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, DistanceIndexPair, Compare> HeapType;
	device_heap = reinterpret_cast<IDeviceHeap*>(malloc(sizeof(HeapType)));
	(*device_heap) = HeapType(
			reinterpret_cast<KeyValuePair<TKey, TValue>*>(storage),
			capacity,
			core::MaxHeapKeyCompare<TKey, TValue>
	);
}

__global__
void PopHeap(IDeviceHeap* device_heap, void* key, void* data) {
	device_heap->pop_internal(key, data);
}

__global__
void IsHeapEmpty(IDeviceHeap* device_heap, bool* empty) {
	*empty = device_heap->empty();
}

__global__
void GetHeapSize(IDeviceHeap* device_heap, int32_t* size) {
	*size = device_heap->size();
}

template<>
class HostHeap<open3d::core::Device::DeviceType::CUDA> : IHostHeap {
public:
	HostHeap(int32_t capacity,
	         const open3d::core::Dtype& key_data_type,
	         const open3d::core::Dtype& value_data_type,
	         const open3d::core::Device& device,
	         HeapType heap_type) :
			key_data_type(key_data_type),
			value_data_type(value_data_type),
			device(device),
			storage(nullptr),
			device_heap(nullptr) {
		dim3 grid_size(1);
		dim3 block_size(1);
		if (key_data_type == open3d::core::Dtype::Float32) {
			if (value_data_type == open3d::core::Dtype::Int32) {
				cudaMalloc(&storage, sizeof(KeyValuePair<int32_t, float>) * capacity);
				switch (heap_type) {
					case HeapType::MIN:
						MakeMinHeap<float, int32_t><<<grid_size, block_size>>>(device_heap, storage, capacity);
						open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("MakeMinHeap() failed.");
						break;
					case HeapType::MAX:
						MakeMaxHeap<float, int32_t><<<grid_size, block_size>>>(device_heap, storage, capacity);
						open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("MakeMaxHeap() failed.");
						break;
					default:
						open3d::utility::LogError("Unsupported heap type, {}.", heap_type);
				}

			}
		} else {
			open3d::utility::LogError("Unsupported Hash key datatype, {}.", key_data_type.ToString());
		}
	}

	~HostHeap() {
		cudaFree(storage);
		cudaFree(device_heap);
	}

	void Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) override {
		auto input_keys_data = reinterpret_cast<const uint8_t*>(input_keys.GetDataPtr());
		auto input_values_data = reinterpret_cast<const uint8_t*>(input_values.GetDataPtr());
		o3c::kernel::CUDALauncher::LaunchGeneralKernel(
				input_keys.GetLength(),
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					device_heap->insert_internal(reinterpret_cast<const void*>(input_keys_data + key_data_type.ByteSize() * workload_idx),
					                             input_values_data + value_data_type.ByteSize() * workload_idx);
				}
		);

	}

	void Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) override {
		output_key = o3c::Tensor({1, 1}, key_data_type, device);
		output_value = o3c::Tensor({1, 1}, value_data_type, device);
		dim3 grid_size(1);
		dim3 block_size(1);
		PopHeap<<<grid_size, block_size>>>(device_heap, output_key.GetDataPtr(), output_value.GetDataPtr());
		open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap Pop() failed.");
	}

	int size() const override {
		int* size_device;
		cudaMalloc(&size_device, sizeof(int));
		int size;
		dim3 grid_size(1);
		dim3 block_size(1);
		GetHeapSize<<<grid_size, block_size>>>(device_heap, size_device);
		open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap size() failed.");
		cudaMemcpy(&size, size_device, sizeof(int), cudaMemcpyDeviceToHost);
		return size;
	};

	bool empty() const override {
		bool* is_empty_device;
		cudaMalloc(&is_empty_device, sizeof(bool));
		bool is_empty;
		dim3 grid_size(1);
		dim3 block_size(1);
		IsHeapEmpty<<<grid_size, block_size>>>(device_heap, is_empty_device);
		open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap empty() failed.");
		cudaMemcpy(&is_empty, is_empty_device, sizeof(bool), cudaMemcpyDeviceToHost);
		return is_empty;
	};
private:
	IDeviceHeap* device_heap;
	const open3d::core::Dtype key_data_type;
	const open3d::core::Dtype value_data_type;
	const open3d::core::Device device;
	void* storage;
};

} // namespace core
} // namespace nnrt