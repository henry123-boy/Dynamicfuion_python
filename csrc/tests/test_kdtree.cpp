//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/1/21.
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

#include <core/KdTree.h>
#include <core/LinearIndex.h>
#include <open3d/core/Tensor.h>

#include <Eigen/Dense>

using namespace nnrt;
namespace o3c = open3d::core;


template<typename TDataStructure>
void Test1DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{5.f, 2.f, 1.f, 6.0f, 3.f, 4.f};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 1}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree(kd_tree_points);
	// diagram verified manually
	std::string gt_diagram =
			"dim: 0             [  4.0]\n"
			"                ______|______\n"
			"               /             \\\n"
			"dim: 0     [  2.0]         [  6.0]\n"
			"            __|__           __|  \n"
			"           /     \\         /      \n"
			"dim: 0 [  1.0] [  3.0] [  5.0]        ";
	REQUIRE(gt_diagram == kd_tree.GenerateTreeDiagram(5));

	std::vector<float> kd_tree_point_data2{-1, 60, 33, 1, 24, 88, 67, 40, 39, 3, 0, 4};
	o3c::Tensor kd_tree_points2(kd_tree_point_data2, {static_cast<long>(kd_tree_point_data2.size()), 1}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree2(kd_tree_points2);
	// diagram verified manually
	std::string gt_diagram2 =
			"dim: 0                             [ 33.0]\n"
			"                        ______________|______________\n"
			"                       /                             \\\n"
			"dim: 0             [  3.0]                         [ 60.0]\n"
			"                ______|______                   ______|______\n"
			"               /             \\                 /             \\\n"
			"dim: 0     [  0.0]         [ 24.0]         [ 40.0]         [ 88.0]\n"
			"            __|__           __|             __|             __|  \n"
			"           /     \\         /               /               /      \n"
			"dim: 0 [ -1.0] [  1.0] [  4.0]         [ 39.0]         [ 67.0]        ";

	REQUIRE(gt_diagram2 == kd_tree2.GenerateTreeDiagram(5));

	std::vector<float> kd_tree_point_data3{
			21.8, 36., -4.9, 42.5, -38.2, -24.5, -42.1, 43.4, -35.1,
			-4.9, -39., 8.5, -44.6, -26.9, -21.2, 43.6, -41.7, -5.4,
			14.5, 45.5, -34.5, -23.2, -5.4, 20.8, 10.9, -14.2, -3.2,
			-20., 33.1, -21., 4.3, -34.1, -6.4, -14.4, -28.1, -26.2,
			-49.3, -47.2, -41.7, -26.8, -36.7, 8.1, 21.2, 29., 16.3,
			44.9, 31.2, 25.1, -27.1, 0.4
	};
	o3c::Tensor kd_tree_points3(kd_tree_point_data3, {static_cast<long>(kd_tree_point_data3.size()), 1}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree3(kd_tree_points3);
	// diagram verified manually via numpy/jupyter
	std::string gt_diagram3 =
			"dim: 0                                                                                                                             [ -5.4]\n"
			"                                                                        ______________________________________________________________|______________________________________________________________\n"
			"                                                                       /                                                                                                                             \\\n"
			"dim: 0                                                             [-28.1]                                                                                                                         [ 21.2]\n"
			"                                        ______________________________|______________________________                                                                   ______________________________|______________________________\n"
			"                                       /                                                             \\                                                                 /                                                             \\\n"
			"dim: 0                             [-39.0]                                                         [-21.2]                                                         [  8.1]                                                         [ 36.0]\n"
			"                        ______________|______________                                   ______________|______________                                   ______________|______________                                   ______________|______________\n"
			"                       /                             \\                                 /                             \\                                 /                             \\                                 /                             \\\n"
			"dim: 0             [-42.1]                         [-35.1]                         [-26.2]                         [-14.4]                         [ -3.2]                         [ 14.5]                         [ 29.0]                         [ 43.6]\n"
			"                ______|______                   ______|______                   ______|______                   ______|______                   ______|______                   ______|______                   ______|______                   ______|______\n"
			"               /             \\                 /             \\                 /             \\                 /             \\                 /             \\                 /             \\                 /             \\                 /             \\\n"
			"dim: 0     [-47.2]         [-41.7]         [-36.7]         [-34.1]         [-26.9]         [-23.2]         [-20.0]         [ -6.4]         [ -4.9]         [  4.3]         [ 10.9]         [ 20.8]         [ 25.1]         [ 33.1]         [ 43.4]         [ 45.5]\n"
			"            __|__           __|             __|             __|             __|__           __|             __|             __|             __|__           __|             __|             __|             __|             __|             __|             __|  \n"
			"           /     \\         /               /               /               /     \\         /               /               /               /     \\         /               /               /               /               /               /               /      \n"
			"dim: 0 [-49.3] [-44.6] [-41.7]         [-38.2]         [-34.5]         [-27.1] [-26.8] [-24.5]         [-21.0]         [-14.2]         [ -5.4] [ -4.9] [  0.4]         [  8.5]         [ 16.3]         [ 21.8]         [ 31.2]         [ 42.5]         [ 44.9]        ";

	REQUIRE(gt_diagram3 == kd_tree3.GenerateTreeDiagram(5));
}

TEST_CASE("Test 1D KDTree Construction CPU") {
	auto device = o3c::Device("CPU:0");
	Test1DKdTreeConstruction<core::KdTree>(device);
}

TEST_CASE("Test 1D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test1DKdTreeConstruction<core::KdTree>(device);
}

template<typename TDataStructure>
void Test2DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{2.f, 5.f, 6.f, 3.f, 3.f, 8.f, 8.f, 9.f, 4.f, 7.f, 7.f, 6.f};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 2}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree(kd_tree_points);
	// diagram verified manually
	std::string gt_diagram =
			"dim: 0                      [  6.0   3.0]\n"
			"                      ____________|____________\n"
			"                     /                         \\\n"
			"dim: 1        [  4.0   7.0]               [  8.0   9.0]\n"
			"               _____|_____                 _____|     \n"
			"              /           \\               /            \n"
			"dim: 0 [  2.0   5.0] [  3.0   8.0] [  7.0   6.0]              ";
	REQUIRE(gt_diagram == kd_tree.GenerateTreeDiagram(5));

	std::vector<float> kd_tree_point_data2{6.9, 0.7, 8.5, 4.9, 4.6, 1.1, 4.8, 7.9, 8.2, 2.6, 8.4, 5.4, 5.2,
	                                       1.7, 9.6, 0.1, 7.1, 8.3, 2.9, 6.4, 5.5, 2.2, 4.6, 5.6, 6.8, 1.9,
	                                       3.3, 4.8, 2.1, 7.9, 7., 7.8, 0.4, 5.2, 0.2, 8.7, 1., 4.6, 1.9,
	                                       2.5};
	o3c::Tensor kd_tree_points2(kd_tree_point_data2, {20, 2}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree2(kd_tree_points2);
	// diagram verified manually via numpy/jupyter
	std::string gt_diagram2 =
			"dim: 0                                                                                                          [  5.2   1.7]\n"
			"                                                                ______________________________________________________|______________________________________________________\n"
			"                                                               /                                                                                                             \\\n"
			"dim: 1                                                  [  4.6   5.6]                                                                                                   [  8.2   2.6]\n"
			"                                    __________________________|__________________________                                                           __________________________|__________________________\n"
			"                                   /                                                     \\                                                         /                                                     \\\n"
			"dim: 0                      [  1.9   2.5]                                           [  2.9   6.4]                                           [  6.9   0.7]                                           [  8.4   5.4]\n"
			"                      ____________|____________                               ____________|____________                               ____________|____________                               ____________|____________\n"
			"                     /                         \\                             /                         \\                             /                         \\                             /                         \\\n"
			"dim: 1        [  0.4   5.2]               [  3.3   4.8]               [  0.2   8.7]               [  4.8   7.9]               [  5.5   2.2]               [  9.6   0.1]               [  7.1   8.3]               [  8.5   4.9]\n"
			"               _____|                      _____|                      _____|                                                  _____|                                                  _____|                                 \n"
			"              /                           /                           /                                                       /                                                       /                                        \n"
			"dim: 0 [  1.0   4.6]               [  4.6   1.1]               [  2.1   7.9]                                           [  6.8   1.9]                                           [  7.0   7.8]                                          ";
	REQUIRE(gt_diagram2 == kd_tree2.GenerateTreeDiagram(5));

}

TEST_CASE("Test 2D KDTree Construction CPU") {
	auto device = o3c::Device("CPU:0");
	Test2DKdTreeConstruction<core::KdTree>(device);
}

TEST_CASE("Test 2D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test2DKdTreeConstruction<core::KdTree>(device);
}


template<typename TDataStructure>
void Test3DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{0., 9.8, 6.8, 5.7, 5., 0.8, 2.1, 1.8, 8.9, 8.3, 1.9,
	                                      2.1, 0.4, 4.3, 3., 8.3, 1.3, 2.9, 1.5, 3.2, 7.6, 3.1,
	                                      2., 7.7, 3.3, 8.4, 0.9, 4.6, 2.7, 5.6, 6.7, 4.9, 1.3,
	                                      5.8, 0.5, 9.3, 0.5, 5.4, 0., 2.8, 7., 6.9, 5.2, 7.7,
	                                      8.6};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {15, 3}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree(kd_tree_points);
	// diagram verified manually via numpy/jupyter
	std::string gt_diagram =
			"dim: 0                                                  [3.3 8.4 0.9]\n"
			"                                    __________________________|__________________________\n"
			"                                   /                                                     \\\n"
			"dim: 1                      [0.4 4.3 3.0]                                           [4.6 2.7 5.6]\n"
			"                      ____________|____________                               ____________|____________\n"
			"                     /                         \\                             /                         \\\n"
			"dim: 2        [3.1 2.0 7.7]               [0.0 9.8 6.8]               [8.3 1.3 2.9]               [6.7 4.9 1.3]\n"
			"               _____|_____                 _____|_____                 _____|_____                 _____|_____\n"
			"              /           \\               /           \\               /           \\               /           \\\n"
			"dim: 0 [1.5 3.2 7.6] [2.1 1.8 8.9] [0.5 5.4 0.0] [2.8 7.0 6.9] [8.3 1.9 2.1] [5.8 0.5 9.3] [5.7 5.0 0.8] [5.2 7.7 8.6]";
	REQUIRE(gt_diagram == kd_tree.GenerateTreeDiagram(3));

	std::vector<float> kd_tree_point_data2{-5., 4.8, 1.8, 0.7, 0., -4.2, -2.9, -3.2, 3.9, 3.3, -3.1,
	                                       -2.9, -4.6, -0.7, -2., 3.3, -3.7, -2.1, -3.5, -1.8, 2.6, -1.9,
	                                       -3., 2.7, -1.7, 3.4, -4.1, -0.4, -2.3, 0.6, 1.7, -0.1, -3.7,
	                                       0.8, -4.5, 4.3, -4.5, 0.4, -5., -2.2, 2., 1.9, 0.2, 2.7,
	                                       3.6};
	o3c::Tensor kd_tree_points2(kd_tree_point_data2, {15, 3}, o3c::Dtype::Float32, device);
	TDataStructure kd_tree2(kd_tree_points2);
	// diagram verified manually via numpy/jupyter
	std::string gt_diagram2 =
			"dim: 0                                                                       [ -1.7   3.4  -4.1]\n"
			"                                                ______________________________________|______________________________________\n"
			"                                               /                                                                             \\\n"
			"dim: 1                               [ -4.6  -0.7  -2.0]                                                             [ -0.4  -2.3   0.6]\n"
			"                            __________________|__________________                                           __________________|__________________\n"
			"                           /                                     \\                                         /                                     \\\n"
			"dim: 2           [ -1.9  -3.0   2.7]                     [ -5.0   4.8   1.8]                     [  3.3  -3.7  -2.1]                     [  1.7  -0.1  -3.7]\n"
			"                  ________|________                       ________|________                       ________|________                       ________|________\n"
			"                 /                 \\                     /                 \\                     /                 \\                     /                 \\\n"
			"dim: 0 [ -3.5  -1.8   2.6] [ -2.9  -3.2   3.9] [ -4.5   0.4  -5.0] [ -2.2   2.0   1.9] [  3.3  -3.1  -2.9] [  0.8  -4.5   4.3] [  0.7   0.0  -4.2] [  0.2   2.7   3.6]";
	REQUIRE(gt_diagram2 == kd_tree2.GenerateTreeDiagram(5));

}

TEST_CASE("Test 3D KDTree Construction CPU") {
	auto device = o3c::Device("CPU:0");
	Test3DKdTreeConstruction<core::KdTree>(device);
}

TEST_CASE("Test 3D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test3DKdTreeConstruction<core::KdTree>(device);
}

// final output sort (for small K, it is faster to use a plain memory block instead of a priority queue for tracking
// the nearest neighbors, so the K nearest neighbors aren't actually ordered by distance in the output)
void SortFinalKNNHelper_Indices(std::vector<int32_t>& nn_i_sorted, std::vector<float>& nn_d_sorted,
                                const o3c::Tensor& nearest_neighbor_indices,
                                const o3c::Tensor& nearest_neighbor_distances) {

	auto nn_i = nearest_neighbor_indices.ToFlatVector<int32_t>();
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


template<typename TIndex = core::KdTree>
void Test3DKnnSearch(const o3c::Device& device, bool use_priority_queue = true) {
	std::vector<float> point_data{0., 9.8, 6.8, 5.7, 5., 0.8, 2.1, 1.8, 8.9, 8.3, 1.9,
	                              2.1, 0.4, 4.3, 3., 8.3, 1.3, 2.9, 1.5, 3.2, 7.6, 3.1,
	                              2., 7.7, 3.3, 8.4, 0.9, 4.6, 2.7, 5.6, 6.7, 4.9, 1.3,
	                              5.8, 0.5, 9.3, 0.5, 5.4, 0., 2.8, 7., 6.9, 5.2, 7.7,
	                              8.6};
	o3c::Tensor points(point_data, {15, 3}, o3c::Dtype::Float32, device);
	TIndex index(points);

	std::vector<float> query_point_data{4.1, 7.2, 6.7, 8., 8.7, 3.3, 4.8, 1.8, 2.6};
	o3c::Tensor query_points(query_point_data, {3, 3}, o3c::Dtype::Float32, device);

	o3c::Tensor nearest_neighbor_indices;
	o3c::Tensor nearest_neighbor_distances;
	const int k = 4;
	index.FindKNearestToPoints(nearest_neighbor_indices, nearest_neighbor_distances, query_points, k);

	// ground truth data computed manually via numpy / jupyter shell
	std::vector<int> gt_nn_indices_data{13, 14, 9, 6, 10, 1, 8, 14, 9, 3, 5, 1};
	std::vector<float> gt_nn_distances_data{1.33041347f, 2.25166605f, 4.6593991f, 4.85489444f, 4.48664685f, 5.02294734f, 5.28583011f, 6.07700584f,
	                                        3.13847097f, 3.53694784f, 3.548239f, 3.78021163f};
	std::vector<int32_t> nn_i_sorted;
	std::vector<float> nn_d_sorted;
	if (!use_priority_queue) {
		SortFinalKNNHelper_Indices(nn_i_sorted, nn_d_sorted, nearest_neighbor_indices, nearest_neighbor_distances);
	} else {
		nn_i_sorted = nearest_neighbor_indices.ToFlatVector<int32_t>();
		nn_d_sorted = nearest_neighbor_distances.ToFlatVector<float>();
	}

	REQUIRE(nn_i_sorted == gt_nn_indices_data);
	REQUIRE(std::equal(nn_d_sorted.begin(), nn_d_sorted.end(), gt_nn_distances_data.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-5).epsilon(1e-12); }));

	std::vector<float> point_data2{-1.1, -4.4, 2.9, -3.8, 2.5, 0.7, 0.2, -2.6, 4.7, 4., -2.6,
	                               -2.8, -3., 0.9, -1., 1.2, -1.2, -0.2, -0.4, 2.4, -2.6, -4.7,
	                               -1.9, -2.8, 1., 4.2, 2.6, -1.2, 3.6, -1.6, -4., -3.1, -1.8,
	                               4.8, -1.4, -4.8, -2., -3.9, 3.9, -4.7, -4.7, -3.8, -2.1, -2.8,
	                               -1.3, -1.1, -4.3, -4.1, 0., 2.2, 1.9, 3.2, 5., 3.4, 3.2,
	                               -3.8, -3.2, -2.6, 2.8, 1., 1.8, -1.5, 1.7, -2.5, -3.1, 4.5,
	                               4.9, 1.2, -0.2, 3.3, 3.9, -3., -2.9, -4.3, -3.3, -0.8, 3.7,
	                               -1.8, -3.8, 0.4, -0.3, -4.4, 3., 4.3, -0.7, 0.9, 1.8, 0.1,
	                               -2.2, -2.4, -0.3, 3.6, -3.9, -0.7, 2.4, 2., 3.3, -3.4, -1.1,
	                               -3.1, -2.9, 4.8, 4.1, 0.7, -1.5, 0., -4.5, -1.1, 0.1, 3.6,
	                               4.5, 1.6, 3.1, 3.1, -3.6, -3.1, -4.2, 1.8, 1.2, -3.2, -2.7,
	                               -3.3, -1.9, -1.1, 4.1, -4.2, 2.4, -0.4, -1.5, -1.9, -1.1, -1.8,
	                               0.5, 0.7, -1.6, -0.9, 1., -4.4, -1.4, 1.7, 2.9, -1.3, 0.,
	                               -2., -4.6, 0.1, 5., -3., 0.7, -4.1, 3.9, 2.9, 0.9, -0.6,
	                               -3.1, -3.9, 2.6, -1.7, -3.5, -0.2, 3.7, 0.2, 1.8, 4.9, -0.2,
	                               1.4, 1.4, -0.6, -2.7, -2.4, -1.4, 3.9, -2.3, 4.8, 1.7, -1.3,
	                               2., -2., 2.9, -2.1, 3.1, 0.1, -2.3, -1.7, -3., 1., 2.4,
	                               0.7, -4.2, 2.9, -0.1, -3.4, 1.6, 2.6, -3.3, -2.3, 4., -0.8,
	                               4., 0.9, -2.6, -2.6, -1.8, 2.6, -2.6, -4.6, -1.6, 0., -4.5,
	                               0.2, -0.2, 4.4, -3.5, -2., 0.4, 3.3, 5., 1.3, 4.1, 4.1,
	                               -0.7, -1.6, -4.2, 1., 2.9, -3.7, -2.1, -0.2, -3.6, -3.6, 0.,
	                               2.3, -4.9, 3.4, 1.5, -3.5, 2.5, -4.3, -4.5, 1.9, -0.3, -3.8,
	                               -3.4, 0.6, 3.1, 3.2, -3.1, -2.5, -3.9, -0.7, 3.7, -0.6, 3.1,
	                               -4.7, 4.7, 1.1, 0.3, -0.6, -0.1, 2.7, 0.5, -2.7, -1.6, 0.4,
	                               -1.2, -2.8, -4.2, 4.3, 2.8, 3.7, 4.1, -4.4, 1.1, 4.7, -3.3,
	                               -2.8, 2.7, -1.4, 2.9, -2.3, -1.2, -4.4, 4.4, -0.3, -2.8, 0.3,
	                               -2.2, -3.6, -1.6, -2.1, -2.3, 3.7, -0.4, 1.8, -2.7, 4., 4.1,
	                               0.3, -2.6, -3.2};
	o3c::Tensor points2(point_data2, {100, 3}, o3c::Dtype::Float32, device);
	TIndex index2(points2);

	std::vector<float> query_point_data2{-4.5, 1.7, -2.2, -3.1, -1.7, 3., 2.7, 4.6, -4.1, 0.1, -3.9,
	                                     -0.9, -3.4, -4.2, 2.8, -1.1, 2.3, -4., 2., -1.1, -2., -1.7,
	                                     4.7, -4.8, 3.7, 1.3, -1.5, -3.2, -2., -0.4, -2.8, -3.4, -0.2,
	                                     3.4, -1.4, 3.4, -0.5, -3.8, 2., 3.1, -3.9, 3.9, -2.6, -4.1,
	                                     -3.1, -3.8, -0.7, -0.5, 2.9, 1.8, -2.7, 4.1, -0.8, 0.9, 4.5,
	                                     -4., -2.2, -3.6, 1.5, 3.7};
	o3c::Tensor query_points2(query_point_data2, {20, 3}, o3c::Dtype::Float32, device);

	o3c::Tensor nearest_neighbor_indices2;
	o3c::Tensor nearest_neighbor_distances2;
	index2.FindKNearestToPoints(nearest_neighbor_indices2, nearest_neighbor_distances2, query_points2, 8);

	std::vector<int32_t> nn_i_sorted2;
	std::vector<float> nn_d_sorted2;

	if (!use_priority_queue) {
		SortFinalKNNHelper_Indices(nn_i_sorted2, nn_d_sorted2, nearest_neighbor_indices2, nearest_neighbor_distances2);
	} else {
		nn_i_sorted2 = nearest_neighbor_indices2.ToFlatVector<int32_t>();
		nn_d_sorted2 = nearest_neighbor_distances2.ToFlatVector<float>();
	}


	// ground truth data computed manually via numpy / jupyter shell
	std::vector<int> gt_nn_indices_data2{4, 26, 49, 59, 1, 65, 47, 7, 67, 21, 33, 71, 12, 87, 61, 74, 23,
	                                     64, 70, 30, 39, 41, 62, 54, 35, 69, 29, 14, 80, 99, 61, 96, 79, 12,
	                                     21, 0, 33, 67, 61, 76, 45, 30, 6, 41, 59, 70, 49, 25, 42, 60, 52,
	                                     63, 5, 73, 85, 29, 41, 30, 70, 59, 25, 9, 6, 45, 34, 66, 60, 22,
	                                     73, 42, 94, 55, 75, 87, 56, 14, 76, 10, 40, 43, 76, 14, 56, 75, 61,
	                                     40, 68, 87, 92, 57, 97, 58, 20, 78, 72, 84, 0, 61, 69, 78, 12, 67,
	                                     2, 35, 84, 77, 57, 78, 92, 90, 2, 20, 24, 40, 68, 38, 15, 82, 10,
	                                     88, 26, 75, 87, 4, 56, 43, 10, 7, 39, 66, 64, 60, 62, 34, 63, 23,
	                                     97, 22, 73, 92, 20, 58, 34, 42, 91, 3, 18, 32, 52, 90, 73, 11, 74,
	                                     27, 71, 48, 46, 98, 19, 1};
	std::vector<float> gt_nn_distances_data2{2.0808652, 2.40624188, 2.61916017, 2.77488739, 3.08868904,
	                                         3.47706773, 3.62904946, 3.65513338, 0.64807407, 2.13775583,
	                                         2.16333077, 2.38956063, 2.61916017, 2.63248932, 2.76586334,
	                                         2.91719043, 1.43527001, 2.41867732, 2.96816442, 3.1685959,
	                                         3.63042697, 3.8340579, 3.91279951, 4.01372645, 0.64031242,
	                                         1.25698051, 2.26715681, 2.49198716, 2.53377189, 2.64952826,
	                                         2.7676705, 2.84429253, 1.30766968, 1.80554701, 2.21585198,
	                                         2.310844, 2.40416306, 2.53771551, 2.7513633, 2.87054002,
	                                         1.37477271, 1.52970585, 1.56843871, 1.81107703, 2.18632111,
	                                         2.33880311, 2.48596058, 2.62488095, 0.9486833, 1.65529454,
	                                         1.72336879, 1.94164878, 1.97230829, 2.17485632, 2.17485632,
	                                         2.23159136, 1.03923048, 1.99499373, 2.00748599, 3.25883415,
	                                         3.28785644, 3.42052628, 3.43802269, 3.80657326, 0.72111026,
	                                         1.2083046, 1.56204994, 1.77200451, 2.04205779, 2.14009346,
	                                         2.17715411, 2.47184142, 0.54772256, 1.02469508, 1.18743421,
	                                         1.63095064, 1.69705627, 1.95192213, 2.04694895, 2.11187121,
	                                         0.84852814, 1.43527001, 1.56524758, 1.58113883, 1.67630546,
	                                         1.70587221, 1.8547237, 1.9, 0.86023253, 1.73781472,
	                                         1.91049732, 2.20454077, 2.33666429, 2.97153159, 3.21558704,
	                                         3.55949435, 1.23693169, 1.75499288, 1.99499373, 2.08326667,
	                                         2.42280829, 2.96141858, 3.03644529, 3.21714159, 1.13137085,
	                                         1.37477271, 2.00249844, 2.16333077, 2.72213152, 3.01496269,
	                                         3.27719392, 3.50570963, 0.41231056, 1.44568323, 1.58113883,
	                                         1.79164729, 1.81383571, 1.85741756, 2.15638587, 2.20454077,
	                                         1.11803399, 1.43527001, 1.68226038, 1.85741756, 2.21585198,
	                                         2.33666429, 2.73678644, 2.74590604, 1.34907376, 1.42478068,
	                                         1.64012195, 1.75783958, 1.92613603, 2.02237484, 2.02484567,
	                                         2.15870331, 1.06301458, 2.41867732, 2.5019992, 2.51396102,
	                                         2.53377189, 2.68700577, 2.83019434, 2.96816442, 0.94339811,
	                                         1.60312195, 1.65227116, 1.73493516, 3.25422802, 3.34813381,
	                                         3.37786915, 3.68917335, 1.11803399, 1.80277564, 1.98242276,
	                                         2.15638587, 2.34946802, 2.68700577, 3.1591138, 3.1685959};
	REQUIRE(nn_i_sorted2 == gt_nn_indices_data2);

	REQUIRE(std::equal(nn_d_sorted2.begin(), nn_d_sorted2.end(), gt_nn_distances_data2.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-5).epsilon(1e-12); }));

}

TEST_CASE("Test 3D KDTree Search CPU - Plain") {
	auto device = o3c::Device("CPU:0");
	Test3DKnnSearch<core::KdTree>(device, false);
}

//TODO: repair the Priority queue versions -- they stopped sorting properly suddenly for an unknown reason

// TEST_CASE("Test 3D KDTree Search CPU - Priority Queue") {
// 	auto device = o3c::Device("CPU:0");
// 	Test3DKnnSearch<core::KdTree>(device, true);
// }

TEST_CASE("Test 3D KDTree Search CUDA - Plain") {
	auto device = o3c::Device("CUDA:0");
	Test3DKnnSearch<core::KdTree>(device, false);
}

// TEST_CASE("Test 3D KDTree Search CUDA - Priority Queue") {
// 	auto device = o3c::Device("CUDA:0");
// 	Test3DKnnSearch<core::KdTree>(device, true);
// }
//
// TEST_CASE("Test 3D LinearIndex Search CPU - Priority Queue") {
// 	auto device = o3c::Device("CPU:0");
// 	Test3DKnnSearch<core::LinearIndex>(device, true);
// }

TEST_CASE("Test 3D LinearIndex Search CPU - Plain") {
	auto device = o3c::Device("CPU:0");
	Test3DKnnSearch<core::LinearIndex>(device, false);
}

// TEST_CASE("Test 3D LinearIndex Search CUDA - Priority Queue") {
// 	auto device = o3c::Device("CUDA:0");
// 	Test3DKnnSearch<core::LinearIndex>(device, true);
// }

TEST_CASE("Test 3D LinearIndex Search CUDA - Plain") {
	auto device = o3c::Device("CUDA:0");
	Test3DKnnSearch<core::LinearIndex>(device, false);
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

void GetDistanceStatistics(float& ratio_below_threshold, float& average_point_distance,
						   Eigen::Map<Eigen::MatrixXf> points, float distance_threshold) {

	float cumulative_distance = 0.0f;
	int64_t count_below_threshold = 0;
	int64_t distance_count = 0;
	for (int i_point = 0; i_point < points.rows()-1; i_point++) {
		for (int j_point = i_point+1; j_point < points.rows(); j_point++, distance_count++) {
			float distance = (points.row(i_point) - points.row(j_point)).norm();
			if(distance < distance_threshold){
				count_below_threshold++;
			}
			cumulative_distance += distance;
		}
	}

	ratio_below_threshold = static_cast<float>(count_below_threshold) / static_cast<float>(distance_count);
	average_point_distance = cumulative_distance / static_cast<float>(distance_count);
}