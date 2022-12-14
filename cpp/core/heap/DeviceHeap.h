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
#include "core/PlatformIndependentQualifiers.h"


namespace nnrt::core{



template<typename TKeyValuePair>
const auto MinHeapKeyCompare = []NNRT_DEVICE_WHEN_CUDACC(const TKeyValuePair& first, const TKeyValuePair& second){
	return first.key > second.key;
};

template<typename TKeyValuePair>
const auto MaxHeapKeyCompare = []NNRT_DEVICE_WHEN_CUDACC(const TKeyValuePair& first, const TKeyValuePair& second){
	return first.key < second.key;
};

class IDeviceHeap {

public:
	virtual NNRT_DEVICE_WHEN_CUDACC ~IDeviceHeap() = default;
	//"Internal" suffix here mainly to avoid strange warnings from CUDA about overrides
	virtual NNRT_DEVICE_WHEN_CUDACC bool InsertInternal(const void* key, const void* value) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC void PopInternal(void* key, void* value) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC bool Empty() const = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC int Size() const = 0;
};


template<typename TElement>
class TypedDeviceHeap : public IDeviceHeap {
public:
	static constexpr size_t key_size = sizeof(TElement::key);
	static constexpr size_t value_size = sizeof(TElement::value);
	typedef decltype(TElement::key) TKey;
	typedef decltype(TElement::value) TValue;

	NNRT_DEVICE_WHEN_CUDACC ~TypedDeviceHeap() override = default;
	virtual NNRT_DEVICE_WHEN_CUDACC bool Insert(TElement element) = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC TElement Pop() = 0;
	virtual NNRT_DEVICE_WHEN_CUDACC TElement& Head() = 0;
	bool NNRT_DEVICE_WHEN_CUDACC InsertInternal(const void* key, const void* value) override{
		return Insert(TElement{*reinterpret_cast<const TKey*>(key), *reinterpret_cast<const TValue*>(value)});
	}
	void NNRT_DEVICE_WHEN_CUDACC PopInternal(void* key, void* value) override {
		TElement& extremum = this->Head();
		*reinterpret_cast<TKey *>(key) = extremum.key;
		*reinterpret_cast<TValue *>(value) = extremum.value;
		this->Pop();
	}
};

template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TComparison>
class DeviceHeap;

} // namespace nnrt::core


