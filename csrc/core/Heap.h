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

#include <open3d/core/Dtype.h>
#include <open3d/core/Tensor.h>



namespace nnrt{
namespace core{

class IHostHeap{
public:
	virtual void Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values) = 0;
	virtual void Pop(open3d::core::Tensor& output_key, open3d::core::Tensor& output_value) = 0;
	virtual int size() const = 0;
	virtual bool empty() const = 0;
};

enum class HeapType : int{
	MIN = 0,
	MAX = 1
};

class Heap {
	Heap(int32_t capacity,
	     const open3d::core::Dtype& key_data_type,
	     const open3d::core::Dtype& value_data_type,
	     const open3d::core::Device& device,
	     HeapType heap_type);
	~Heap() = default;
	void Insert(const open3d::core::Tensor& input_keys, const open3d::core::Tensor& input_values);
	open3d::core::Tensor Pop();
	int size() const;
	bool empty() const;
private:
	std::shared_ptr<IHostHeap> host_heap;


};

template<open3d::core::Device::DeviceType TDeviceType>
class HostHeap;


} // namespace core
} // namespace nnrt
