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

#include <core/CPU/DeviceHeapCPU.h>
#include <core/Heap.h>
#include <open3d/core/Tensor.h>


#include "tests/test_utils/test_utils.hpp"

using namespace nnrt;
namespace o3c = open3d::core;

TEST_CASE("Test Device Heap CPU") {
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

TEST_CASE("Test Host Heap CPU") {
	const int queue_capacity = 40;

	auto device = o3c::Device("cpu:0");
	auto key_data_type = o3c::Dtype::Float32;
	auto value_data_type = o3c::Dtype::Int32;
	std::vector<float> keys1_data{50.0f, 40.0f, 20.0f, 100.0f};
	o3c::Tensor keys1(keys1_data, {4}, key_data_type, device);
	std::vector<int32_t> values1_data{4, 3, 2, 6};
	o3c::Tensor values1(values1_data, {4}, value_data_type, device);

	core::Heap heap(queue_capacity, key_data_type, value_data_type, device, core::HeapType::MIN);
	REQUIRE(heap.Empty() == true);

	heap.Insert(keys1, values1);
	REQUIRE(heap.Size() == 4);
	REQUIRE(heap.Empty() == false);

	o3c::Tensor min_key1, min_value1;

	heap.Pop(min_key1, min_value1);
	REQUIRE(min_key1[0].Item<float>() == 20.0f);
	REQUIRE(min_value1[0].Item<int32_t>() == 2);

	std::vector<float> keys2_data{60.0f, 10.0f};
	o3c::Tensor keys2(keys2_data, {2}, key_data_type, device);
	std::vector<int32_t> values2_data{5, 1};
	o3c::Tensor values2(values2_data, {2}, value_data_type, device);
	heap.Insert(keys2, values2);
	REQUIRE(heap.Size() == 5);

	std::vector<float> expected_output_key_data{10.f, 40.f, 50.f, 60.f, 100.f};
	std::vector<int32_t> expected_output_value_data{1, 3, 4, 5, 6};

	for(int i_item = 0; i_item < 5; i_item++){
		o3c::Tensor min_key, min_value;
		heap.Pop(min_key, min_value);
		REQUIRE(min_key[0].Item<float>() == expected_output_key_data[i_item]);
		REQUIRE(min_value[0].Item<int32_t>() == expected_output_value_data[i_item]);
	}
	REQUIRE(heap.Size() == 0);
	REQUIRE(heap.Empty() == true);
}


TEST_CASE("Test Host Heap CUDA") {
	const int queue_capacity = 40;

	auto device = o3c::Device("cuda:0");
	auto cpu = o3c::Device("cpu:0");
	auto key_data_type = o3c::Dtype::Float32;
	auto value_data_type = o3c::Dtype::Int32;
	std::vector<float> keys1_data{50.0f, 40.0f, 20.0f, 100.0f};
	o3c::Tensor keys1(keys1_data, {4}, key_data_type, device);
	std::vector<int32_t> values1_data{4, 3, 2, 6};
	o3c::Tensor values1(values1_data, {4}, value_data_type, device);

	core::Heap heap(queue_capacity, key_data_type, value_data_type, device, core::HeapType::MIN);
	REQUIRE(heap.Empty() == true);

	heap.Insert(keys1, values1);
	REQUIRE(heap.Size() == 4);
	REQUIRE(heap.Empty() == false);

	o3c::Tensor min_key1, min_value1;
	heap.Pop(min_key1, min_value1);

	REQUIRE(min_key1.To(cpu)[0].Item<float>() == 20.0f);
	REQUIRE(min_value1.To(cpu)[0].Item<int32_t>() == 2);
	REQUIRE(heap.Size() == 3);

	std::vector<float> keys2_data{60.0f, 10.0f};
	o3c::Tensor keys2(keys2_data, {2}, key_data_type, device);
	std::vector<int32_t> values2_data{5, 1};
	o3c::Tensor values2(values2_data, {2}, value_data_type, device);

	heap.Insert(keys2, values2);
	REQUIRE(heap.Size() == 5);

	std::vector<float> expected_output_key_data{10.f, 40.f, 50.f, 60.f, 100.f};
	std::vector<int32_t> expected_output_value_data{1, 3, 4, 5, 6};

	for(int i_item = 0; i_item < 5; i_item++){
		o3c::Tensor min_key, min_value;
		heap.Pop(min_key, min_value);
		REQUIRE(min_key.To(cpu)[0].To(cpu).Item<float>() == expected_output_key_data[i_item]);
		REQUIRE(min_value.To(cpu)[0].To(cpu).Item<int32_t>() == expected_output_value_data[i_item]);
	}
	REQUIRE(heap.Size() == 0);
	REQUIRE(heap.Empty() == true);
}


