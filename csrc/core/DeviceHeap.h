//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/28/21.
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
#include <open3d/core/Device.h>
#include <open3d/core/Dtype.h>
#include "core/KeyValuePair.h"
#include "utility/PlatformIndependence.h"


namespace nnrt{
namespace core{



template<typename TKey, typename TValue>
const auto MinHeapKeyCompare = []NNRT_DEVICE_WHEN_CUDACC(const KeyValuePair<TKey, TValue>& first, const KeyValuePair<TKey, TValue>& second){
	return first.key > second.key;
};

template<typename TKey, typename TValue>
const auto MaxHeapKeyCompare = []NNRT_DEVICE_WHEN_CUDACC(const KeyValuePair<TKey, TValue>& first, const KeyValuePair<TKey, TValue>& second){
	return first.key < second.key;
};

class IDeviceHeap {

public:
	virtual ~IDeviceHeap() = default;
	virtual NNRT_DEVICE_WHEN_CUDACC bool insert_internal(const void* key, const void* value) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC void pop_internal(void* key, void* value) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC bool empty() const = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC int size() const = 0;
};



template<typename TElement>
class TypedDeviceHeap : public IDeviceHeap {
public:
	static constexpr size_t key_size = sizeof(TElement::key);
	static constexpr size_t value_size = sizeof(TElement::value);
	typedef decltype(TElement::key) TKey;
	typedef decltype(TElement::value) TValue;

	NNRT_DEVICE_WHEN_CUDACC ~TypedDeviceHeap() override = default;
	virtual NNRT_DEVICE_WHEN_CUDACC bool insert(TElement element) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC TElement pop() = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC TElement& head() = 0;
	bool NNRT_DEVICE_WHEN_CUDACC insert_internal(const void* key, const void* value) override{
		return insert(TElement{*reinterpret_cast<const TKey *>(key), *reinterpret_cast<const TValue *>(value)});
	}
	void NNRT_DEVICE_WHEN_CUDACC pop_internal(void* key, void* value) override {
		TElement& extremum = this->head();
		*reinterpret_cast<TKey *>(key) = extremum.key;
		*reinterpret_cast<TValue *>(value) = extremum.value;
		this->pop();
	}
};

template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TComparison>
class DeviceHeap;



} // namespace core
} // namespace nnrt


