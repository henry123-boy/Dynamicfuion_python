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


void Test1DKdTreeConstruction(const o3c::Device& device){
	std::vector<float> kd_tree_point_data{5.f, 2.f, 1.f, 6.0f, 3.f, 4.f};

	o3c::Tensor kd_tree_points(kd_tree_point_data, {6, 1}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree(kd_tree_points);
	std::string diagram = kd_tree.GenerateTreeDiagram(5);
	std::string gt_diagram =
			"dim: 0             [  4.0]\n"
			"                ______|______\n"
			"               /             \\\n"
			"dim: 0     [  2.0]         [  6.0]\n"
			"            __|__           __|  \n"
			"           /     \\         /      \n"
			"dim: 0 [  1.0] [  3.0] [  5.0]        ";
	REQUIRE(diagram == gt_diagram);

	std::vector<float> kd_tree_point_data2{-1, 60, 33, 1, 24, 88, 67, 40, 39, 3, 0, 4};
	o3c::Tensor kd_tree_points2(kd_tree_point_data2, {static_cast<long>(kd_tree_point_data2.size()), 1}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree2(kd_tree_points2);
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
			21.8,  36. ,  -4.9,  42.5, -38.2, -24.5, -42.1,  43.4, -35.1,
			-4.9, -39. ,   8.5, -44.6, -26.9, -21.2,  43.6, -41.7,  -5.4,
			14.5,  45.5, -34.5, -23.2,  -5.4,  20.8,  10.9, -14.2,  -3.2,
			-20. ,  33.1, -21. ,   4.3, -34.1,  -6.4, -14.4, -28.1, -26.2,
			-49.3, -47.2, -41.7, -26.8, -36.7,   8.1,  21.2,  29. ,  16.3,
			44.9,  31.2,  25.1, -27.1,   0.4
	};
	o3c::Tensor kd_tree_points3(kd_tree_point_data3, {static_cast<long>(kd_tree_point_data3.size()), 1}, o3c::Dtype::Float32, device);
	core::KdTree kd_tree3(kd_tree_points3);
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