//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/30/21.
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
#include "core/DeviceHeap.h"


namespace nnrt {
namespace core {


template<>
class HostHeap<open3d::core::Device::DeviceType::CUDA> : public IHostHeap {
public:
	HostHeap(int32_t capacity,
	         const open3d::core::Dtype& key_data_type,
	         const open3d::core::Dtype& value_data_type,
	         const open3d::core::Device& device,
	         HeapType heap_type);

	~HostHeap();

	void Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) override;

	void Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) override;
	int Size() const override;

	bool Empty() const override;
private:
	IDeviceHeap* device_heap;
	const open3d::core::Dtype key_data_type;
	const open3d::core::Dtype value_data_type;
	const open3d::core::Device device;
	void* storage;
};

} // namespace core
} // namespace nnrt