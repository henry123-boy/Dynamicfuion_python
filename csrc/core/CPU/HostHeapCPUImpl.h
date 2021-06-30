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
#include "open3d/core/kernel/CPULauncher.h"

namespace o3c = open3d::core;

namespace nnrt {
namespace core {



template<>
HostHeap<open3d::core::Device::DeviceType::CPU> ::HostHeap(int32_t capacity,
	         const open3d::core::Dtype& key_data_type,
	         const open3d::core::Dtype& value_data_type,
	         const open3d::core::Device& device,
	         HeapType heap_type) :
			key_data_type(key_data_type),
			value_data_type(value_data_type),
			device(device){

		if (key_data_type == open3d::core::Dtype::Float32) {
			if (value_data_type == open3d::core::Dtype::Int32) {
				storage = malloc(sizeof(KeyValuePair<int32_t, float>) * capacity);
				switch (heap_type) {

					case HeapType::MIN:

						device_heap = std::make_shared<DeviceHeap<
								open3d::core::Device::DeviceType::CPU,
								KeyValuePair<float, int32_t>,
								decltype(MinHeapKeyCompare<float, int32_t>)
						>>(
								reinterpret_cast<KeyValuePair<float,int32_t>*>(storage),
								capacity,
								MinHeapKeyCompare<float, int32_t>
						);
						break;
					case HeapType::MAX:
						device_heap = std::make_shared<DeviceHeap<
								open3d::core::Device::DeviceType::CPU,
								KeyValuePair<float, int32_t>,
								decltype(MaxHeapKeyCompare<float, int32_t>)
						>>(
								reinterpret_cast<KeyValuePair<float, int32_t>*>(storage),
								capacity,
								MaxHeapKeyCompare<float, int32_t>
						);
					default:
						open3d::utility::LogError("Unsupported heap type, {}.", heap_type);
				}

			}
		} else {
			open3d::utility::LogError("Unsupported Hash key datatype, {}.", key_data_type.ToString());
		}
	}

	~HostHeap() {
		free(storage);
	}

	void Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) override {
		auto input_keys_data = reinterpret_cast<const uint8_t*>(input_keys.GetDataPtr());
		auto input_values_data = reinterpret_cast<const uint8_t*>(input_values.GetDataPtr());
		o3c::kernel::CPULauncher::LaunchGeneralKernel(
				input_keys.GetLength(),
				[=] (int64_t workload_idx){
					device_heap->insert_internal(reinterpret_cast<const void *>(input_keys_data + key_data_type.ByteSize() * workload_idx),
												 input_values_data + value_data_type.ByteSize() * workload_idx);
				}
		);

	}

	void Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) override {
		output_key = o3c::Tensor({1,1}, key_data_type, device);
		output_value = o3c::Tensor({1,1}, value_data_type, device);
		device_heap->pop_internal(output_key.GetDataPtr(), output_value.GetDataPtr());
	}

	int size() const override {
		return device_heap->size();
	};

	bool empty() const override {
		return device_heap->empty();
	};
private:
	std::shared_ptr<IDeviceHeap> device_heap;
	const open3d::core::Dtype key_data_type;
	const open3d::core::Dtype value_data_type;
	const open3d::core::Device device;
	void* storage;
};

} // namespace core
} // namespace nnrt
