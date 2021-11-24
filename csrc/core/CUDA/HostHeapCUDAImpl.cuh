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
#include "core/CUDA/HostHeapCUDA.h"
#include "core/CUDA/DeviceHeapCUDA.cuh"
#include <open3d/core/CUDAUtils.h>


namespace nnrt::core {

__global__
void InsertIntoHeap(IDeviceHeap* device_heap, const uint8_t* input_keys_data, const uint8_t* input_values_data,
					const int64_t key_byte_size, const int64_t value_byte_size, const int64_t count){

	for(int64_t i_pair = 0; i_pair < count; i_pair++){
		device_heap->InsertInternal(reinterpret_cast<const void*>(input_keys_data + key_byte_size * i_pair),
		                    reinterpret_cast<const void*>(input_values_data + value_byte_size * i_pair));
	}
}

template<typename TKey, typename TValue>
__global__
void MakeMinHeap(IDeviceHeap* device_heap, void* storage, const int capacity) {
	typedef core::KeyValuePair<TKey, TValue> DistanceIndexPair;
	typedef decltype(core::MinHeapKeyCompare<TKey, TValue>) Compare;

	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, DistanceIndexPair, Compare> HT;
	HT local(
			reinterpret_cast<KeyValuePair<TKey, TValue>*>(storage),
			capacity,
			core::MinHeapKeyCompare<TKey, TValue>
	);
	memcpy(device_heap,&local, sizeof(HT));
}

template<typename TKey, typename TValue>
__global__
void MakeMaxHeap(IDeviceHeap* device_heap, void* storage, const int capacity) {
	typedef core::KeyValuePair<TKey, TValue> DistanceIndexPair;
	typedef decltype(core::MaxHeapKeyCompare<TKey, TValue>) Compare;

	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, DistanceIndexPair, Compare> HT;
	HT local(
			reinterpret_cast<KeyValuePair<TKey, TValue>*>(storage),
			capacity,
			core::MaxHeapKeyCompare<TKey, TValue>
	);
	memcpy(device_heap,&local, sizeof(HT));
}

__global__
void PopHeap(IDeviceHeap* device_heap, void* key, void* data) {
	device_heap->PopInternal(key, data);
}

__global__
void IsHeapEmpty(IDeviceHeap* device_heap, bool* empty) {
	*empty = device_heap->Empty();
}

__global__
void GetHeapSize(IDeviceHeap* device_heap, int32_t* size) {
	*size = device_heap->Size();
}

template<typename TKey, typename TValue, typename TCompare>
core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, core::KeyValuePair<TKey, TValue>, TCompare>
        MakeHeap(void* storage, int capacity, TCompare compare) {
	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, core::KeyValuePair<TKey, TValue>, TCompare> HT;
	auto ret = HT(
			reinterpret_cast<KeyValuePair<TKey, TValue>*>(storage),
			capacity,
			compare
	);
	return ret;
}

template<typename TKey, typename TValue, typename TCompare>
inline
size_t GetHeapSize(TCompare&& compare){
	typedef core::DeviceHeap<open3d::core::Device::DeviceType::CUDA, core::KeyValuePair<TKey, TValue>, TCompare> HT;
	return sizeof(HT);
}


HostHeap<open3d::core::Device::DeviceType::CUDA>::HostHeap(int32_t capacity,
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
					OPEN3D_CUDA_CHECK(cudaMalloc(&device_heap, GetHeapSize<float, int32_t>(core::MinHeapKeyCompare<float, int32_t>)));
					MakeMinHeap<float,int32_t><<<grid_size, block_size>>>(device_heap, storage, capacity);
					break;
				case HeapType::MAX:
					OPEN3D_CUDA_CHECK(cudaMalloc(&device_heap, GetHeapSize<float, int32_t>(core::MinHeapKeyCompare<float, int32_t>)));
					MakeMaxHeap<float,int32_t><<<grid_size, block_size>>>(device_heap, storage, capacity);
					break;
				default:
					open3d::utility::LogError("Unsupported heap type, {}.", heap_type);
			}

		}
	} else {
		open3d::utility::LogError("Unsupported Hash key datatype, {}.", key_data_type.ToString());
	}
}

HostHeap<open3d::core::Device::DeviceType::CUDA>::~HostHeap() {
	cudaFree(storage);
	cudaFree(device_heap);
}

void HostHeap<open3d::core::Device::DeviceType::CUDA>::Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) {
	input_keys.AssertDtype(this->key_data_type);
	input_values.AssertDtype(this->value_data_type);
	input_keys.AssertDevice(this->device);
	input_values.AssertDevice(this->device);
	auto input_keys_data = reinterpret_cast<const uint8_t*>(input_keys.GetDataPtr());
	auto input_values_data = reinterpret_cast<const uint8_t*>(input_values.GetDataPtr());
	dim3 grid_size(1);
	dim3 block_size(1);
	InsertIntoHeap<<<grid_size, block_size>>>(device_heap, input_keys_data, input_values_data,
	                                          key_data_type.ByteSize(), value_data_type.ByteSize(), input_keys.GetLength());
	open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap Pop() failed.");
}

void HostHeap<open3d::core::Device::DeviceType::CUDA>::Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) {
	output_key = o3c::Tensor({1}, key_data_type, device);
	output_value = o3c::Tensor({1}, value_data_type, device);
	dim3 grid_size(1);
	dim3 block_size(1);
	PopHeap<<<grid_size, block_size>>>(device_heap, output_key.GetDataPtr(), output_value.GetDataPtr());
	open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap Pop() failed.");
}

int HostHeap<open3d::core::Device::DeviceType::CUDA>::Size() const {
	int* size_device;
	cudaMalloc(&size_device, sizeof(int));
	int size;
	dim3 grid_size(1);
	dim3 block_size(1);
	GetHeapSize<<<grid_size, block_size>>>(device_heap, size_device);
	open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap Size() failed.");
	cudaMemcpy(&size, size_device, sizeof(int), cudaMemcpyDeviceToHost);
	return size;
};

bool HostHeap<open3d::core::Device::DeviceType::CUDA>::Empty() const {
	bool* is_empty_device;
	cudaMalloc(&is_empty_device, sizeof(bool));
	bool is_empty;
	dim3 grid_size(1);
	dim3 block_size(1);
	IsHeapEmpty<<<grid_size, block_size>>>(device_heap, is_empty_device);
	open3d::core::OPEN3D_GET_LAST_CUDA_ERROR("Heap Empty() failed.");
	cudaMemcpy(&is_empty, is_empty_device, sizeof(bool), cudaMemcpyDeviceToHost);
	return is_empty;
};

} // namespace nnrt::core