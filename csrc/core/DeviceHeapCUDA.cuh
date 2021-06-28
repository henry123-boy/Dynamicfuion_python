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
namespace cuda {
template<typename T>
__device__ void swap(T& x, T& y) {
	T t = x;
	x = y;
	y = t;
}
} // namespace cuda
namespace heap {


/**
 * \brief moves an element down the heap until all children are smaller / larger than the element
 * dependent on the passed-in compare function
 * \tparam TCompare
 * \tparam RandomAccessIterator
 * \param array
 * \param begin
 * \param length
 * \param compare
 */
template<typename RandomAccessIterator, typename TCompare>
__device__ void
sift_down(RandomAccessIterator array, size_t begin, size_t length, TCompare compare) {
	while (2 * begin + 1 < length) {
		size_t left = 2 * begin + 1;
		size_t right = 2 * begin + 2;
		size_t extremum = begin;
		if ((left < length) && compare(array[extremum], array[left])) extremum = left;

		if ((right < length) && compare(array[extremum], array[right])) extremum = right;

		if (extremum != begin) {
			cuda::swap(array[begin], array[extremum]);
			begin = extremum;
		} else return;
	}
}


template<typename RandomAccessIterator, typename TCompare>
__device__ void
make_heap(RandomAccessIterator begin, size_t length, TCompare compare) {
	int i = static_cast<int>(length / 2 - 1);
	while (i >= 0) {
		sift_down(begin, i, length, compare);
		i--;
	}
}


} // namespace heap


template<typename TElement, typename TCompare>
class DeviceHeap<o3c::Device::DeviceType::CUDA, TElement, TCompare> : IDeviceHeap<TElement> {
public:
	__device__ DeviceHeap(TElement* data, int capacity, TCompare compare) :
			data(data), cursor(capacity), capacity(capacity), compare(compare), size(0) {}

	__device__ bool insert(TElement element) override {
		if(size >= capacity) return false;
		// fill in from the back, moving toward the front
		cursor--;
		data[cursor] = element;
		heap::sift_down(data, cursor, capacity, compare);
		size++;
		return true;
	}

	__device__ TElement pop() override {
		assert(size > 0);
		TElement extremum = data[cursor];
		cursor++;
		size--;
		heap::make_heap(data + cursor, size, compare);
		return extremum;
	}

	__device__ bool empty() override {
		return size == 0;
	}

private:
	TElement* const data;
	int cursor;
	const int capacity;
	int size;
	const TCompare compare;
};


} // namespace core
} // namespace nnrt