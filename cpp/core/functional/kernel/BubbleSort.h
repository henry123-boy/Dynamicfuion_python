//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/13/22.
//  Copyright (c) 2022 Gregory Kramida
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

#include "core/platform_independence/Qualifiers.h"

namespace nnrt::core::functional::kernel {

// We only use Bubble sort for tiny thread-local arrays (n < 20). In these cases, we care more about warp divergence than computational complexity.

template<typename TElement>
NNRT_DEVICE_WHEN_CUDACC
inline void SwapElements(TElement* array, int index_a, int index_b) {
	TElement temp = array[index_a];
	array[index_a] = array[index_b];
	array[index_b] = temp;
}

template<typename TElement>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSort(TElement* array, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			if (array[j_element] > array[j_element + 1]) {
				SwapElements(array, j_element, j_element + 1);
			}
		}
	}
}

template<typename TElement, class = std::enable_if_t<std::is_signed<TElement>::value>>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSort_PositiveFirst(TElement* array, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			auto& el_0 = array[j_element];
			auto& el_1 = array[j_element + 1];
			if (el_1 >= 0) {
				if (el_0 < 0 || el_0 > el_1) {
					SwapElements(array, j_element, j_element + 1);
				}
			} else {
				if (el_0 < 0 && el_0 > el_1) {
					SwapElements(array, j_element, j_element + 1);
				}
			}
		}
	}
}

template<typename TElement>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSort_Descending(TElement* array, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			if (array[j_element] < array[j_element + 1]) {
				SwapElements(array, j_element, j_element + 1);
			}
		}
	}
}

template<typename TElement, class = std::enable_if_t<std::is_signed<TElement>::value>>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSort_Descending_NegativeFirst(TElement* array, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			auto& el_0 = array[j_element];
			auto& el_1 = array[j_element + 1];
			if (el_1 < 0) {
				if (el_0 >= 0 || el_0 < el_1) {
					SwapElements(array, j_element, j_element + 1);
				}
			} else {
				if (el_0 >= 0 && el_0 < el_1) {
					SwapElements(array, j_element, j_element + 1);
				}
			}
		}
	}
}


template<typename TValue, typename TKey>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSortByKey(TValue* values, TKey* keys, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			if (keys[j_element] > keys[j_element + 1]) {
				SwapElements(keys, j_element, j_element + 1);
				SwapElements(values, j_element, j_element + 1);
			}
		}
	}
}

template<typename TValue, typename TKey, class = std::enable_if_t<std::is_signed<TKey>::value>>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSortByKey_PositiveFirst(TValue* values, TKey* keys, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			auto& key_0 = keys[j_element];
			auto& key_1 = keys[j_element + 1];
			if (key_1 >= 0) {
				if (key_0 < 0 || key_0 > key_1) {
					SwapElements(keys, j_element, j_element + 1);
					SwapElements(values, j_element, j_element + 1);
				}
			} else {
				if (key_0 < 0 && key_0 > key_1) {
					SwapElements(keys, j_element, j_element + 1);
					SwapElements(values, j_element, j_element + 1);
				}
			}
		}
	}
}

template<typename TValue, typename TKey>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSortByKey_Descending(TValue* values, TKey* keys, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			if (keys[j_element] < keys[j_element + 1]) {
				SwapElements(keys, j_element, j_element + 1);
				SwapElements(values, j_element, j_element + 1);
			}
		}
	}
}

template<typename TValue, typename TKey, class = std::enable_if_t<std::is_signed<TKey>::value>>
NNRT_DEVICE_WHEN_CUDACC
inline void BubbleSortByKey_Descending_NegativeFirst(TValue* values, TKey* keys, int element_count) {
	for (int i_element = 0; i_element < element_count - 1; i_element++) {
		for (int j_element = 0; j_element < element_count - i_element - 1; j_element++) {
			auto& key_0 = keys[j_element];
			auto& key_1 = keys[j_element + 1];
			if (key_1 < 0) {
				if (key_0 >= 0 || key_0 < key_1) {
					SwapElements(keys, j_element, j_element + 1);
					SwapElements(values, j_element, j_element + 1);
				}
			} else {
				if (key_0 >= 0 && key_0 < key_1) {
					SwapElements(keys, j_element, j_element + 1);
					SwapElements(values, j_element, j_element + 1);
				}
			}
		}
	}
}

} // namespace nnrt::core::functional::kernel