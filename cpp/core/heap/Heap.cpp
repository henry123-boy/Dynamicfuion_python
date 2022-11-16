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
#include "Heap.h"
#include "core/heap/CPU/HostHeapCPU.h"
#include "core/heap/CUDA/HostHeapCUDA.h"

namespace o3c = open3d::core;

namespace nnrt::core {


Heap::Heap(const int32_t capacity,
           const open3d::core::Dtype& key_data_type,
           const open3d::core::Dtype& value_data_type,
           const open3d::core::Device& device,
           HeapType heap_type) {
	switch (device.GetType()) {

		case open3d::core::Device::DeviceType::CPU:
			this->host_heap = std::make_shared<HostHeap<o3c::Device::DeviceType::CPU>>(
					capacity, key_data_type, value_data_type, device, heap_type);
			break;
		case open3d::core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			this->host_heap = std::make_shared<HostHeap<o3c::Device::DeviceType::CUDA>>(
					capacity, key_data_type, value_data_type, device, heap_type);
#else
			open3d::utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			open3d::utility::LogError("Unimplemented device");
			break;
	}

}

void Heap::Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) {
	this->host_heap->Insert(input_keys, input_values);
}

void Heap::Pop(open3d::core::Tensor& key, open3d::core::Tensor& value) {
	this->host_heap->Pop(key, value);
}

int Heap::Size() const {
	return this->host_heap->Size();
}

bool Heap::Empty() const {
	return this->host_heap->Empty();
}

pybind11::tuple Heap::Pop() {
	o3c::Tensor key, value;
	this->Pop(key, value);
	return pybind11::make_tuple(key, value);
}


} // namespace nnrt::core