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

#include "core/PlatformIndependence.h"

namespace nnrt::rendering::kernel {

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
	// Bubble sort. We only use it for tiny thread-local arrays (n < 8); in this
	// regime we care more about warp divergence than computational complexity.
	for (int i_element = 0; i_element < element_count - 1; ++i_element) {
		for (int j_element = 0; j_element < element_count - i_element - 1; ++j_element) {
			if (array[j_element + 1] < array[j_element]) {
				SwapElements(array, j_element, j_element + 1);
			}
		}
	}
}

} // namespace nnrt::rendering::kernel