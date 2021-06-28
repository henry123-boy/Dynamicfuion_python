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

#include "core/DeviceHeap.h"

namespace o3c = open3d::core;

namespace nnrt {
namespace core {

template<typename TElement, typename TCompare>
class DeviceHeap<o3c::Device::DeviceType::CPU, TElement, TCompare> : IDeviceHeap<TElement> {
public:
	DeviceHeap(TElement* data, int capacity, TCompare compare) :
			data(data), capacity(capacity), compare(compare), size(0) {}

	bool insert(TElement element) override {
		if (size >= capacity) return false;
		data[size] = element;
		std::push_heap(data, data + size, compare);
		size++;
		return true;

	}

	TElement pop() override {
		if (size > 0) {
			std::pop_heap(data, data + size, compare);
			TElement extremum = data[size];
			size--;
			return extremum;
		} else {
			open3d::utility::LogError("Trying to pop from an empty heap.");
		}
	}

	bool empty() override {
		return size == 0;
	}

private:
	TElement* const data;
	const int capacity;
	int size;
	const TCompare compare;
};

} // namespace core
} // namespace nnrt