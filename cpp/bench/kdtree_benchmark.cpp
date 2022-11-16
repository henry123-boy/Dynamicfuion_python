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


struct BenchmarkResult {
	double brute_force_time;
	double kd_tree_time;
};

BenchmarkResult BenchmarkKnnTreeIndexForDevice(const o3c::Device& device, int query_point_count, int reference_point_count,
                                               int k, bool sorted_output = false, bool compare_results = false);


int main() {
	const int query_point_count = 1000000;
	const int reference_point_count = 10000;
	const int k = 8;
	const int run_count = 10;
	o3c::Device cpu("CPU:0");
	o3c::Device cuda("CUDA:0");

	double total_bf_time = 0.0;
	double total_kd_time = 0.0;

	for (int i_run = 0; i_run < run_count; i_run++) {
		auto result = BenchmarkKnnTreeIndexForDevice(cuda, query_point_count, reference_point_count, k, false, false);
		total_bf_time += result.brute_force_time;
		total_kd_time += result.kd_tree_time;
	}
	std::cout << "Average brute-force search time over " << run_count << " runs: " << total_bf_time / run_count << " s" << std::endl;
	std::cout << "Average KD-tree search time over " << run_count << " runs: " << total_kd_time / run_count << " s" << std::endl;
}

void PrepareData(o3c::Tensor& query_points, o3c::Tensor& reference_points,
                 const int query_point_count, const int reference_point_count, const o3c::Device& device) {
	const int point_dimension_count = 3;
	std::random_device random_device;
	std::default_random_engine random_engine(random_device());
	std::uniform_real_distribution<float> uniform_distribution(-500.f, 500.f);

	std::vector<float> query_point_data(query_point_count * point_dimension_count);
	for (float& entry: query_point_data) {
		entry = uniform_distribution(random_engine);
	}
	query_points = o3c::Tensor(query_point_data, {query_point_count, point_dimension_count}, o3c::Dtype::Float32, device);

	std::vector<float> reference_point_data(reference_point_count * point_dimension_count);
	for (float& entry: reference_point_data) {
		entry = uniform_distribution(random_engine);
	}
	reference_points = o3c::Tensor(reference_point_data, {reference_point_count, point_dimension_count}, o3c::Dtype::Float32, device);
}

// final output sort (for small K, it is faster to use a plain memory block instead of a priority queue for tracking
// the nearest neighbors, so the K nearest neighbors aren't actually ordered by distance in the output)
template<typename TIndexElement>
void SortFinalKNNHelper_Indices(std::vector<TIndexElement>& nn_i_sorted, std::vector<float>& nn_d_sorted,
                                const o3c::Tensor& nearest_neighbor_indices,
                                const o3c::Tensor& nearest_neighbor_distances) {

	auto nn_i = nearest_neighbor_indices.ToFlatVector<TIndexElement>();
	nn_i_sorted.resize(nn_i.size());
	auto nn_d = nearest_neighbor_distances.ToFlatVector<float>();
	nn_d_sorted.resize(nn_d.size());
	const int k = static_cast<int>(nearest_neighbor_indices.GetShape(1));
	for (int i_query_point = 0; i_query_point < nearest_neighbor_indices.GetShape(0); i_query_point++) {
		std::vector<int> idx(k);
		iota(idx.begin(), idx.end(), 0);
		const int offset = i_query_point * k;
		stable_sort(idx.begin(), idx.end(),
		            [&nn_d, &offset](int i1, int i2) {
			            return nn_d[offset + i1] < nn_d[offset + i2];
		            });
		for (int i_neighbor = 0; i_neighbor < k; i_neighbor++) {
			nn_i_sorted[offset + i_neighbor] = nn_i[offset + idx[i_neighbor]];
			nn_d_sorted[offset + i_neighbor] = nn_d[offset + idx[i_neighbor]];
		}
	}
}

BenchmarkResult BenchmarkKnnTreeIndexForDevice(const o3c::Device& device, const int query_point_count,
                                               const int reference_point_count, const int k, bool sorted_output, bool compare_results) {
	using namespace std::chrono;
	const int point_dimension_count = 3;

	std::cout << "Testing on device `" << device.ToString() << "`." << std::endl;

	std::cout << "Preparing data..." << std::endl;
	o3c::Tensor query_points, reference_points;
	PrepareData(query_points, reference_points, query_point_count, reference_point_count, device);
	std::cout << "Data prepared." << std::endl;

	std::cout << "Constructing KD Tree..." << std::endl;
	auto start = high_resolution_clock::now();
	nnrt::core::KdTree kd_tree(reference_points);
	auto end = high_resolution_clock::now();
	std::cout << "Finished constructing KD Tree. Time: " << duration_cast<duration<double>>(end - start).count() << " seconds." << std::endl;

	nnrt::core::LinearIndex linear_index(reference_points);

	o3c::Tensor neighbor_indices_bf, neighbor_distances_bf;
	std::cout << "Searching linear index for query points' KNN (brute force, multi-threaded)..." << std::endl;
	start = high_resolution_clock::now();
	linear_index.FindKNearestToPoints(neighbor_indices_bf, neighbor_distances_bf, query_points, k, sorted_output);
	end = high_resolution_clock::now();
	double brute_force_time = duration_cast<duration<double>>(end - start).count();
	std::cout << "Finished searching linear index.\nBrute force KNN time: " << duration_cast<duration<double>>(end - start).count() << " seconds."
	          << std::endl;

	o3c::Tensor neighbor_indices_kdtree, neighbor_distances_kdtree;
	std::cout << "Searching KD Tree for query points' KNN... (multi-threaded)" << std::endl;
	start = high_resolution_clock::now();
	kd_tree.FindKNearestToPoints(neighbor_indices_kdtree, neighbor_distances_kdtree, query_points, k, sorted_output);
	end = high_resolution_clock::now();
	double kd_tree_time = duration_cast<duration<double>>(end - start).count();
	std::cout << "Finished searching KD Tree.\nKD Tree KNN time: " << kd_tree_time << " seconds." << std::endl;

	if (compare_results) {
		std::cout << "Comparing results...." << std::endl;
		std::vector<int32_t> nn_i_sorted_bf, nn_i_sorted_kdtree;
		std::vector<float> nn_d_sorted_bf, nn_d_sorted_kdtree;
		if (sorted_output) {
			nn_i_sorted_bf = neighbor_indices_bf.ToFlatVector<int32_t>();
			nn_d_sorted_bf = neighbor_distances_bf.ToFlatVector<float>();
			nn_i_sorted_kdtree = neighbor_indices_kdtree.ToFlatVector<int32_t>();
			nn_d_sorted_kdtree = neighbor_distances_kdtree.ToFlatVector<float>();
		} else {
			SortFinalKNNHelper_Indices(nn_i_sorted_bf, nn_d_sorted_bf, neighbor_indices_bf, neighbor_distances_bf);
			SortFinalKNNHelper_Indices(nn_i_sorted_kdtree, nn_d_sorted_kdtree, neighbor_indices_kdtree, neighbor_distances_kdtree);
		}

		std::cout << "Indices match: " << (nn_i_sorted_bf == nn_i_sorted_kdtree ? "true" : "false") << std::endl;
		for (int i = 0; i < nn_i_sorted_bf.size(); i++) {
			if (nn_i_sorted_bf[i] != nn_i_sorted_kdtree[i]) {
				std::cout << "Mismatch at index " << i << ":" << std::endl << "KD Tree: "
				          << nn_i_sorted_kdtree[i] << " vs. LI: " << nn_i_sorted_bf[i] << std::endl;
			}
		}
		std::cout << "Distances match: " << (std::equal(nn_d_sorted_bf.begin(), nn_d_sorted_bf.end(), nn_d_sorted_kdtree.begin(),
		                                                [](float a, float b) { return std::abs(a - b) <= 1e-5; }) ? "true" : "false") << std::endl;
	}
	return {brute_force_time, kd_tree_time};
}

// final output sort (for small K, it is faster to use a plain memory block instead of a priority queue for tracking
// the nearest neighbors, so the K nearest neighbors aren't actually ordered by distance in the output)
void SortFinalKNNHelper_Points(std::vector<float>& nn_p_sorted, std::vector<float>& nn_d_sorted,
                               const o3c::Tensor& nearest_neighbors,
                               const o3c::Tensor& nearest_neighbor_distances) {

	auto nn_p = nearest_neighbors.ToFlatVector<float>();
	nn_p_sorted.resize(nn_p.size());
	auto nn_d = nearest_neighbor_distances.ToFlatVector<float>();
	nn_d_sorted.resize(nn_d.size());
	const int k = static_cast<int>(nearest_neighbors.GetShape(1));
	for (int i_query_point = 0; i_query_point < nearest_neighbors.GetShape(0); i_query_point++) {
		std::vector<int> idx(k);
		iota(idx.begin(), idx.end(), 0);
		const int offset = i_query_point * k;
		stable_sort(idx.begin(), idx.end(),
		            [&nn_d, &offset](int i1, int i2) {
			            return nn_d[offset + i1] < nn_d[offset + i2];
		            });
		for (int i_neighbor = 0; i_neighbor < k; i_neighbor++) {
			nn_p_sorted[(offset + i_neighbor) * 3 + 0] = nn_p[(offset + idx[i_neighbor]) * 3 + 0];
			nn_p_sorted[(offset + i_neighbor) * 3 + 1] = nn_p[(offset + idx[i_neighbor]) * 3 + 1];
			nn_p_sorted[(offset + i_neighbor) * 3 + 2] = nn_p[(offset + idx[i_neighbor]) * 3 + 2];
			nn_d_sorted[offset + i_neighbor] = nn_d[offset + idx[i_neighbor]];
		}
	}
}