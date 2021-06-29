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
#include "test_main.hpp"

#include <core/DeviceHeapCPU.h>

#include "tests/test_utils/test_utils.hpp"

using namespace nnrt;

TEST_CASE("Test Heap CPU") {
	typedef core::KeyValuePair<float, int32_t> DistanceIndexPair;
	typedef decltype(core::MinHeapKeyCompare<float, int32_t>) Compare;

	const int queue_capacity = 40;
	DistanceIndexPair queue_data[queue_capacity];
	core::DeviceHeap<open3d::core::Device::DeviceType::CPU, DistanceIndexPair, Compare> heap(queue_data, queue_capacity, core::MinHeapKeyCompare<float, int32_t>);

	DistanceIndexPair queue_data_insert_order[6] = {
			DistanceIndexPair{50.0f, 4},
			DistanceIndexPair{40.0f, 3},
			DistanceIndexPair{20.0f, 2},
			DistanceIndexPair{100.0f, 6},
			DistanceIndexPair{60.0f, 5},
			DistanceIndexPair{10.0f, 1}
	};

	const int first_batch_size = 4;
	for(int i_item = 0; i_item < first_batch_size; i_item++){
		heap.insert(queue_data_insert_order[i_item]);
	}
	auto min_pair1 = heap.pop();
	REQUIRE(min_pair1.key == 20.0f);
	REQUIRE(min_pair1.value == 2);
	for(int i_item = first_batch_size; i_item < 6; i_item++){
		heap.insert(queue_data_insert_order[i_item]);
	}
	DistanceIndexPair expected_output_order[5] = {
			DistanceIndexPair{10.0f, 1},
			DistanceIndexPair{40.0f, 3},
			DistanceIndexPair{50.0f, 4},
			DistanceIndexPair{60.0f, 5},
			DistanceIndexPair{100.0f, 6}
	};

	for(int i_item = 0; i_item < 5; i_item++){
		auto current_head = heap.pop();
		REQUIRE(current_head.key == expected_output_order[i_item].key);
		REQUIRE(current_head.value == expected_output_order[i_item].value);
	}

}