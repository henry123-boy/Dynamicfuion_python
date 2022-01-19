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
#include "core/CPU/HostHeapCPU.h"
#include "core/CPU/DeviceHeapCPU.h"

namespace o3c = open3d::core;

namespace nnrt::core {

HostHeap<open3d::core::Device::DeviceType::CPU>::
HostHeap(int32_t capacity,
         const open3d::core::Dtype& key_data_type,
         const open3d::core::Dtype& value_data_type,
         const open3d::core::Device& device,
         HeapType heap_type) :
		key_data_type(key_data_type),
		value_data_type(value_data_type),
		device(device) {

	if (key_data_type == open3d::core::Dtype::Float32) {
		if (value_data_type == open3d::core::Dtype::Int32) {
			storage = malloc(sizeof(KeyValuePair<int32_t, float>) * capacity);
			switch (heap_type) {

				case HeapType::MIN:

					device_heap = std::make_shared<DeviceHeap<
							open3d::core::Device::DeviceType::CPU,
							KeyValuePair<float, int32_t>,
							decltype(MinHeapKeyCompare<KeyValuePair<float, int32_t>>)
					>>(
							reinterpret_cast<KeyValuePair<float, int32_t>*>(storage),
							capacity,
							MinHeapKeyCompare<KeyValuePair<float, int32_t>>
					);
					break;
				case HeapType::MAX:
					device_heap = std::make_shared<DeviceHeap<
							open3d::core::Device::DeviceType::CPU,
							KeyValuePair<float, int32_t>,
							decltype(MaxHeapKeyCompare<KeyValuePair<float, int32_t>>)
					>>(
							reinterpret_cast<KeyValuePair<float, int32_t>*>(storage),
							capacity,
							MaxHeapKeyCompare<KeyValuePair<float, int32_t>>
					);
				default:
					open3d::utility::LogError("Unsupported heap type, {}.", heap_type);
			}

		}
	} else {
		open3d::utility::LogError("Unsupported Hash key datatype, {}.", key_data_type.ToString());
	}
}

HostHeap<open3d::core::Device::DeviceType::CPU>::~HostHeap() {
	free(storage);
}

void HostHeap<open3d::core::Device::DeviceType::CPU>::Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) {
	o3c::AssertTensorDtype(input_keys, this->key_data_type);
	o3c::AssertTensorDtype(input_values, this->value_data_type);
	o3c::AssertTensorDevice(input_keys, this->device);
	o3c::AssertTensorDevice(input_values, this->device);
	auto input_keys_data = reinterpret_cast<const uint8_t*>(input_keys.GetDataPtr());
	auto input_values_data = reinterpret_cast<const uint8_t*>(input_values.GetDataPtr());
	for(int64_t i_pair = 0; i_pair < input_keys.GetLength(); i_pair++){
		device_heap->InsertInternal(reinterpret_cast<const void*>(input_keys_data + key_data_type.ByteSize() * i_pair),
		                    input_values_data + value_data_type.ByteSize() * i_pair);
	}
}

void HostHeap<open3d::core::Device::DeviceType::CPU>::Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) {
	output_key = o3c::Tensor({1}, key_data_type, device);
	output_value = o3c::Tensor({1}, value_data_type, device);
	device_heap->PopInternal(output_key.GetDataPtr(), output_value.GetDataPtr());
}

int HostHeap<open3d::core::Device::DeviceType::CPU>::Size() const {
	return device_heap->Size();
};

bool HostHeap<open3d::core::Device::DeviceType::CPU>::Empty() const {
	return device_heap->Empty();
};

} // namespace nnrt::core
