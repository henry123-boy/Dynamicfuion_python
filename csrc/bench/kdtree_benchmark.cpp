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

void BenchmarkForDevice(const o3c::Device& device, const int query_point_count, const int point_count, const int k, bool compare_results = false);

int main() {
	const int query_point_count = 300000;
	const int point_count = 30000;
	const int k = 8;
	// o3c::Device cpu("CPU:0");
	// BenchmarkForDevice(cpu, query_point_count, point_count, k);
	o3c::Device cuda("CUDA:0");
	BenchmarkForDevice(cuda, query_point_count, point_count, k, false);
}

// final output sort (linear index / brute force KNN doesn't even use a priority queue,
// so the K nearest neighbors aren't actually ordered by distance)
void SortFinalKNNHelper(std::vector<int32_t>& nn_i_sorted, std::vector<float>& nn_sd_sorted,
                        const o3c::Tensor& nearest_neighbor_indices,
                        const o3c::Tensor& squared_distances) {

	auto nn_i = nearest_neighbor_indices.ToFlatVector<int32_t>();
	nn_i_sorted.resize(nn_i.size());
	auto nn_sd = squared_distances.ToFlatVector<float>();
	nn_sd_sorted.resize(nn_sd.size());
	const int k = static_cast<int>(nearest_neighbor_indices.GetShape(1));
	for (int i_query_point = 0; i_query_point < nearest_neighbor_indices.GetShape(0); i_query_point++) {
		std::vector<int> idx(k);
		iota(idx.begin(), idx.end(), 0);
		const int offset = i_query_point * k;
		stable_sort(idx.begin(), idx.end(),
		            [&nn_sd, &offset](int i1, int i2) {
			            return nn_sd[offset + i1] < nn_sd[offset + i2];
		            });
		for (int i_neighbor = 0; i_neighbor < k; i_neighbor++) {
			nn_i_sorted[offset + i_neighbor] = nn_i[offset + idx[i_neighbor]];
			nn_sd_sorted[offset + i_neighbor] = nn_sd[offset + idx[i_neighbor]];
		}
	}
}

void BenchmarkForDevice(const o3c::Device& device, const int query_point_count, const int point_count, const int k, bool compare_results) {
	using namespace std::chrono;
	const int point_dimension_count = 3;

	std::cout << "Testing on device `" << device.ToString() << "`." << std::endl;

	std::random_device random_device;
	std::default_random_engine random_engine(random_device());
	std::uniform_real_distribution<float> uniform_distribution(-500.f, 500.f);

	std::vector<float> query_point_data(query_point_count * point_dimension_count);
	for (float& entry: query_point_data) {
		entry = uniform_distribution(random_engine);
	}
	o3c::Tensor query_points(query_point_data, {query_point_count, point_dimension_count}, o3c::Dtype::Float32, device);

	std::vector<float> point_data(point_count * point_dimension_count);
	for (float& entry: point_data) {
		entry = uniform_distribution(random_engine);
	}
	o3c::Tensor points(point_data, {point_count, point_dimension_count}, o3c::Dtype::Float32, device);

	std::cout << "Data prepared." << std::endl;


	std::cout << "Constructing KD Tree..." << std::endl;
	auto start = high_resolution_clock::now();
	nnrt::core::KdTree kd_tree(points);
	auto end = high_resolution_clock::now();
	std::cout << "Finished constructing KD Tree. Time: " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;

	nnrt::core::LinearIndex linear_index(points);

	o3c::Tensor nearest_neighbor_indices_bf, squared_distances_bf;
	std::cout << "Searching linear index for query points' KNN (brute force, multi-threaded)..." << std::endl;
	start = high_resolution_clock::now();
	linear_index.FindKNearestToPoints(nearest_neighbor_indices_bf, squared_distances_bf, query_points, k);
	end = high_resolution_clock::now();
	std::cout << "Finished searching linear index. Time: " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;

	o3c::Tensor nearest_neighbor_indices_kdtree, squared_distances_kdtree;
	std::cout << "Searching KD Tree for query points' KNN... (multi-threaded)" << std::endl;
	start = high_resolution_clock::now();
	kd_tree.FindKNearestToPoints(nearest_neighbor_indices_kdtree, squared_distances_kdtree, query_points, k);
	end = high_resolution_clock::now();
	std::cout << "Finished searching KD Tree. Time: " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;

	if (compare_results) {
		std::cout << "Comparing results...." << std::endl;
		std::vector<int32_t> nn_i_sorted;
		std::vector<float> nn_sd_sorted;
		SortFinalKNNHelper(nn_i_sorted, nn_sd_sorted, nearest_neighbor_indices_bf, squared_distances_bf);
		auto nni_kdtree = nearest_neighbor_indices_kdtree.ToFlatVector<int32_t>();
		std::cout << "Indices match: " << (nn_i_sorted == nni_kdtree ? "true" : "false") << std::endl;
		for (int i = 0; i < nn_i_sorted.size(); i++) {
			if (nn_i_sorted[i] != nni_kdtree[i]) {
				std::cout << "Mismatch at index " << i << ":" << std::endl << "KD Tree: "
				          << nni_kdtree[i] << " vs. LI: " << nn_i_sorted[i] << std::endl;
			}
		}
		auto squared_distances_kdtree_data = squared_distances_kdtree.ToFlatVector<float>();
		std::cout << "Distances match: " << (std::equal(nn_sd_sorted.begin(), nn_sd_sorted.end(), squared_distances_kdtree_data.begin(),
		                                                [](float a, float b) { return std::abs(a - b) <= 1e-5; }) ? "true" : "false") << std::endl;
	}
}