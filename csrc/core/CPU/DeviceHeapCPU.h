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

namespace nnrt::core {

template<typename TElement, typename TCompare>
class DeviceHeap<o3c::Device::DeviceType::CPU, TElement, TCompare> : public TypedDeviceHeap<TElement> {
public:
	DeviceHeap(TElement* data, int capacity, TCompare compare) :
			data(data), capacity(capacity), compare(compare), _size(0) {}

	bool Insert(TElement element) override {
		if (_size >= capacity) return false;
		data[_size] = element;
		_size++;
		std::push_heap(data, data + _size, compare);
		return true;
	}

	TElement Pop() override {
		if (_size > 0) {
			std::pop_heap(data, data + _size, compare);
			TElement extremum = data[_size-1];
			_size--;
			return extremum;
		} else {
			open3d::utility::LogError("Trying to Pop from an empty heap.");
		}
	}

	TElement& Head() override{
		if (_size > 0) {
			return data[0];
		} else {
			open3d::utility::LogError("Trying to Pop from an empty heap.");
		}
	}

	bool Empty() const override {
		return _size == 0;
	}

	int Size() const override{
		return _size;
	}


private:
	TElement* const data;
	const int capacity;
	int _size;
	const TCompare compare;
};



} // namespace nnrt::core