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
//__DEBUG
#include <iostream>

#include "test_main.hpp"

#include <core/KdTree.h>
#include <open3d/core/Tensor.h>

using namespace nnrt;
namespace o3c = open3d::core;


void Test1DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{5.f, 2.f, 1.f, 6.0f, 3.f, 4.f};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 1}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree(kd_tree_points);
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
	core::KdTree kd_tree2(kd_tree_points2);
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

	REQUIRE(gt_diagram2 == kd_tree2.GenerateTreeDiagram());

	std::vector<float> kd_tree_point_data3{
			21.8, 36., -4.9, 42.5, -38.2, -24.5, -42.1, 43.4, -35.1,
			-4.9, -39., 8.5, -44.6, -26.9, -21.2, 43.6, -41.7, -5.4,
			14.5, 45.5, -34.5, -23.2, -5.4, 20.8, 10.9, -14.2, -3.2,
			-20., 33.1, -21., 4.3, -34.1, -6.4, -14.4, -28.1, -26.2,
			-49.3, -47.2, -41.7, -26.8, -36.7, 8.1, 21.2, 29., 16.3,
			44.9, 31.2, 25.1, -27.1, 0.4
	};
	o3c::Tensor kd_tree_points3(kd_tree_point_data3, {static_cast<long>(kd_tree_point_data3.size()), 1}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree3(kd_tree_points3);
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
	Test1DKdTreeConstruction(device);
}

TEST_CASE("Test 1D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test1DKdTreeConstruction(device);
}

void Test2DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{2.f, 5.f, 6.f, 3.f, 3.f, 8.f, 8.f, 9.f, 4.f, 7.f, 7.f, 6.f};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 2}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree(kd_tree_points);
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
	core::KdTree kd_tree2(kd_tree_points2);
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
	Test2DKdTreeConstruction(device);
}

TEST_CASE("Test 2D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test2DKdTreeConstruction(device);
}


void Test3DKdTreeConstruction(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{0., 9.8, 6.8, 5.7, 5., 0.8, 2.1, 1.8, 8.9, 8.3, 1.9,
	                                      2.1, 0.4, 4.3, 3., 8.3, 1.3, 2.9, 1.5, 3.2, 7.6, 3.1,
	                                      2., 7.7, 3.3, 8.4, 0.9, 4.6, 2.7, 5.6, 6.7, 4.9, 1.3,
	                                      5.8, 0.5, 9.3, 0.5, 5.4, 0., 2.8, 7., 6.9, 5.2, 7.7,
	                                      8.6};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {15, 3}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree(kd_tree_points);
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
	core::KdTree kd_tree2(kd_tree_points2);
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
	Test3DKdTreeConstruction(device);
}

TEST_CASE("Test 3D KDTree Construction CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test3DKdTreeConstruction(device);
}

void Test3DKdTreeSearch(const o3c::Device& device) {
	std::vector<float> kd_tree_point_data{0., 9.8, 6.8, 5.7, 5., 0.8, 2.1, 1.8, 8.9, 8.3, 1.9,
	                                      2.1, 0.4, 4.3, 3., 8.3, 1.3, 2.9, 1.5, 3.2, 7.6, 3.1,
	                                      2., 7.7, 3.3, 8.4, 0.9, 4.6, 2.7, 5.6, 6.7, 4.9, 1.3,
	                                      5.8, 0.5, 9.3, 0.5, 5.4, 0., 2.8, 7., 6.9, 5.2, 7.7,
	                                      8.6};
	o3c::Tensor kd_tree_points(kd_tree_point_data, {15, 3}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree(kd_tree_points);

	std::vector<float> query_point_data{4.1, 7.2, 6.7, 8., 8.7, 3.3, 4.8, 1.8, 2.6};
	o3c::Tensor query_points(query_point_data, {3, 3}, o3c::Dtype::Float32, device);

	o3c::Tensor nearest_neighbor_indices;
	o3c::Tensor squared_distances;
	kd_tree.FindKNearestToPoints(nearest_neighbor_indices, squared_distances, query_points, 4);

	// ground truth data computed manually via numpy / jupyter shell
	std::vector<int> gt_nn_indices_data{13, 14, 9, 6, 10, 1, 8, 14, 9, 3, 5, 1};
	std::vector<float> gt_nn_square_distances_data{1.77f, 5.07f, 21.71f, 23.57f, 20.13f, 25.23f, 27.94f, 36.93f, 9.85f, 12.51f, 12.59f, 14.29f};

	REQUIRE(nearest_neighbor_indices.ToFlatVector<int32_t>() == gt_nn_indices_data);
	auto nn_square_distance_data = squared_distances.ToFlatVector<float>();
	REQUIRE(std::equal(nn_square_distance_data.begin(), nn_square_distance_data.end(), gt_nn_square_distances_data.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-5).epsilon(1e-12); }));

	std::vector<float> kd_tree_point_data2{-1.1, -4.4, 2.9, -3.8, 2.5, 0.7, 0.2, -2.6, 4.7, 4., -2.6,
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
	o3c::Tensor kd_tree_points2(kd_tree_point_data2, {100, 3}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree2(kd_tree_points2);

	std::vector<float> query_point_data2{-4.5, 1.7, -2.2, -3.1, -1.7, 3., 2.7, 4.6, -4.1, 0.1, -3.9,
	                                     -0.9, -3.4, -4.2, 2.8, -1.1, 2.3, -4., 2., -1.1, -2., -1.7,
	                                     4.7, -4.8, 3.7, 1.3, -1.5, -3.2, -2., -0.4, -2.8, -3.4, -0.2,
	                                     3.4, -1.4, 3.4, -0.5, -3.8, 2., 3.1, -3.9, 3.9, -2.6, -4.1,
	                                     -3.1, -3.8, -0.7, -0.5, 2.9, 1.8, -2.7, 4.1, -0.8, 0.9, 4.5,
	                                     -4., -2.2, -3.6, 1.5, 3.7};
	o3c::Tensor query_points2(query_point_data2, {20, 3}, o3c::Dtype::Float32, device);

	o3c::Tensor nearest_neighbor_indices2;
	o3c::Tensor squared_distances2;
	kd_tree2.FindKNearestToPoints(nearest_neighbor_indices2, squared_distances2, query_points2, 8);

	// ground truth data computed manually via numpy / jupyter shell
	std::vector<int> gt_nn_indices_data2{ 4, 26, 49, 59,  1, 65, 47,  7, 67, 21, 33, 71, 12, 87, 61, 74, 23,
	                                     64, 70, 30, 39, 41, 62, 54, 35, 69, 29, 14, 80, 99, 61, 96, 79, 12,
	                                     21,  0, 33, 67, 61, 76, 45, 30,  6, 41, 59, 70, 49, 25, 42, 60, 52,
	                                     63,  5, 73, 85, 29, 41, 30, 70, 59, 25,  9,  6, 45, 34, 66, 60, 22,
	                                     73, 42, 94, 55, 75, 87, 56, 14, 76, 10, 40, 43, 76, 14, 56, 75, 61,
	                                     40, 68, 87, 92, 57, 97, 58, 20, 78, 72, 84,  0, 61, 69, 78, 12, 67,
	                                     2, 35, 84, 77, 57, 78, 92, 90,  2, 20, 24, 40, 68, 38, 15, 82, 10,
	                                     88, 26, 75, 87,  4, 56, 43, 10,  7, 39, 66, 64, 60, 62, 34, 63, 23,
	                                     97, 22, 73, 92, 20, 58, 34, 42, 91,  3, 18, 32, 52, 90, 73, 11, 74,
	                                     27, 71, 48, 46, 98, 19,  1};
	std::vector<float> gt_nn_square_distances_data2{4.33,  5.79,  6.86,  7.7 ,  9.54, 12.09, 13.17, 13.36,  0.42,
	                                                4.57,  4.68,  5.71,  6.86,  6.93,  7.65,  8.51,  2.06,  5.85,
	                                                8.81, 10.04, 13.18, 14.7 , 15.31, 16.11,  0.41,  1.58,  5.14,
	                                                6.21,  6.42,  7.02,  7.66,  8.09,  1.71,  3.26,  4.91,  5.34,
	                                                5.78,  6.44,  7.57,  8.24,  1.89,  2.34,  2.46,  3.28,  4.78,
	                                                5.47,  6.18,  6.89,  0.9 ,  2.74,  2.97,  3.77,  3.89,  4.73,
	                                                4.73,  4.98,  1.08,  3.98,  4.03, 10.62, 10.81, 11.7 , 11.82,
	                                                14.49,  0.52,  1.46,  2.44,  3.14,  4.17,  4.58,  4.74,  6.11,
	                                                0.3 ,  1.05,  1.41,  2.66,  2.88,  3.81,  4.19,  4.46,  0.72,
	                                                2.06,  2.45,  2.5 ,  2.81,  2.91,  3.44,  3.61,  0.74,  3.02,
	                                                3.65,  4.86,  5.46,  8.83, 10.34, 12.67,  1.53,  3.08,  3.98,
	                                                4.34,  5.87,  8.77,  9.22, 10.35,  1.28,  1.89,  4.01,  4.68,
	                                                7.41,  9.09, 10.74, 12.29,  0.17,  2.09,  2.5 ,  3.21,  3.29,
	                                                3.45,  4.65,  4.86,  1.25,  2.06,  2.83,  3.45,  4.91,  5.46,
	                                                7.49,  7.54,  1.82,  2.03,  2.69,  3.09,  3.71,  4.09,  4.1 ,
	                                                4.66,  1.13,  5.85,  6.26,  6.32,  6.42,  7.22,  8.01,  8.81,
	                                                0.89,  2.57,  2.73,  3.01, 10.59, 11.21, 11.41, 13.61,  1.25,
	                                                3.25,  3.93,  4.65,  5.52,  7.22,  9.98, 10.04};
	REQUIRE(nearest_neighbor_indices2.ToFlatVector<int32_t>() == gt_nn_indices_data2);
	auto nn_square_distance_data2 = squared_distances2.ToFlatVector<float>();
	REQUIRE(std::equal(nn_square_distance_data2.begin(), nn_square_distance_data2.end(), gt_nn_square_distances_data2.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-5).epsilon(1e-12); }));

}

TEST_CASE("Test 3D KDTree Search CPU") {
	auto device = o3c::Device("CPU:0");
	Test3DKdTreeSearch(device);
}

TEST_CASE("Test 3D KDTree Search CUDA") {
	auto device = o3c::Device("CUDA:0");
	Test3DKdTreeSearch(device);
}