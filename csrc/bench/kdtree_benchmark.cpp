//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/8/22.
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
#include <iostream>
#include <random>
#include <chrono>

#include <core/KdTree.h>
#include <core/LinearIndex.h>
#include <open3d/core/Tensor.h>

namespace o3c = open3d::core;

void BenchmarkForDevice(const o3c::Device& device, int query_point_count, int point_count, int k);

int main() {
	const int query_point_count = 300000;
	const int point_count = 3000;
	const int k = 8;
	// o3c::Device cpu("CPU:0");
	// BenchmarkForDevice(cpu, query_point_count, point_count, k);
	o3c::Device cuda("CUDA:0");
	BenchmarkForDevice(cuda, query_point_count, point_count, k);
}

void BenchmarkForDevice(const o3c::Device& device, const int query_point_count, const int point_count, const int k) {
	using namespace std::chrono;
	const int point_dimension_count = 3;

	std::cout << "Testing on device `" << device.ToString() << "`." << std::endl;

	std::random_device random_device;
	std::default_random_engine random_engine(random_device());
	std::uniform_real_distribution<float> uniform_distribution(-500.f,500.f);

	std::vector<float> query_point_data(query_point_count * point_dimension_count);
	for(float & entry : query_point_data){
		entry = uniform_distribution(random_engine);
	}
	o3c::Tensor query_points(query_point_data, {query_point_count, point_dimension_count}, o3c::Dtype::Float32, device);

	std::vector<float> point_data(point_count * point_dimension_count);
	for(float& entry : point_data){
		entry = uniform_distribution(random_engine);
	}
	o3c::Tensor points(point_data, {point_count, point_dimension_count}, o3c::Dtype::Float32, device);

	std::cout << "Data prepared." << std::endl;


	std::cout << "Constructing KD Tree..." << std::endl;
	auto start = high_resolution_clock::now();
	nnrt::core::KdTree kd_tree(points);
	auto end = high_resolution_clock::now();
	std::cout << "Finished constructing KD Tree. Time: " << duration_cast<duration<double>>(end-start).count() << " seconds." << std::endl;

	nnrt::core::LinearIndex linear_index(points);

	o3c::Tensor nearest_neighbor_indices, squared_distances;

	std::cout << "Searching linear index for query points' KNN (brute force, multi-threaded)..." << std::endl;
	start = high_resolution_clock::now();
	linear_index.FindKNearestToPoints(nearest_neighbor_indices, squared_distances, query_points, k);
	end = high_resolution_clock::now();
	std::cout << "Finished searching linear index. Time: " << duration_cast<duration<double>>(end-start).count() << " seconds." << std::endl;
	std::cout << "Searching KD Tree for query points' KNN... (multi-threaded)" << std::endl;
	start = high_resolution_clock::now();
	kd_tree.FindKNearestToPoints(nearest_neighbor_indices, squared_distances, query_points, k);
	end = high_resolution_clock::now();
	std::cout << "Finished searching KD Tree. Time: " << duration_cast<duration<double>>(end-start).count() << " seconds." << std::endl;


}