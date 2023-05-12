//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/27/22.
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
// standard library
#include <cmath>
// 3rd party
#include <open3d/core/Tensor.h>
// test framework
#include "test_main.hpp"
// code being tested
#include "geometry/GraphWarpField.h"
#include "core/functional/Sorting.h"

namespace o3c = open3d::core;
namespace ngeom = nnrt::geometry;


void TestGraphWarpFieldConstructor(const o3c::Device& device) {
	std::vector<float> nodes_data{0.f, 0.f, 0.f,
	                              0.04f, 0.f, 0.f,
	                              0.f, 0.04f, 0.f};
	o3c::Tensor nodes(nodes_data, {3, 3}, o3c::Dtype::Float32, device);
	std::vector<int> edges_data{1, 2, -1, -1,
	                            2, -1, -1, -1,
	                            -1, -1, -1, -1};
	o3c::Tensor edges(edges_data, {3, 4}, o3c::Dtype::Int32, device);

	std::vector<float> edge_weight_data{0.6f, 0.2f, 0.f, 0.f,
	                                    0.8f, 0.0f, 0.f, 0.f,
	                                    0.0f, 0.0f, 0.f, 0.f,};
	o3c::Tensor edge_weights(edge_weight_data, {3, 4}, o3c::Dtype::Float32, device);
	std::vector<int> clusters_data{0, 0, 0};
	o3c::Tensor clusters(clusters_data, {3}, o3c::Dtype::Int32, device);

	ngeom::WarpField gwf(nodes);
	REQUIRE(gwf.nodes.GetShape(0) == 3);
	REQUIRE(gwf.nodes.GetShape(1) == 3);


	ngeom::PlanarGraphWarpField pgwf(nodes, edges, edge_weights, clusters);
	REQUIRE(pgwf.nodes.GetShape(0) == 3);
	REQUIRE(pgwf.nodes.GetShape(1) == 3);
	REQUIRE(pgwf.nodes.ToFlatVector<float>() == nodes.ToFlatVector<float>());
	REQUIRE(pgwf.edges.ToFlatVector<int>() == edges.ToFlatVector<int>());
	REQUIRE(pgwf.edge_weights.value().get().ToFlatVector<float>() == edge_weights.ToFlatVector<float>());
	REQUIRE(pgwf.clusters.value().get().ToFlatVector<int>() == clusters.ToFlatVector<int>());

}

TEST_CASE("Test Planar Graph Warp Field Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestGraphWarpFieldConstructor(device);
}

TEST_CASE("Test Planar Graph Warp Field Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestGraphWarpFieldConstructor(device);
}

void TestHierarchicalGraphWarpFieldConstructor(const o3c::Device& device) {
	//@formatter:off
	std::vector<float> node_data{
			// 2, 3
			2.11, 3.2, 1,  //  0
			2.32, 3.7, 1,  //  1 <--- winner
			2.66, 3.35, 1, //  2
			// 3, 3
			3.36, 3.7, 1,  //  3 <--- winner
			// 2, 2
			2.31, 2.75, 1, //  4 <--- winner
			2.41, 2.2, 1,  //  5
			2.71, 2.7, 1,  //  6
			// 3, 2
			3.31, 2.8, 1,  //  7
			3.71, 2.8, 1,  //  8
			3.16, 2.3, 1,  //  9
			3.56, 2.3, 1,  // 10 <--- winner
			// 4, 3
			4.21, 3.7, 1,  // 11 <--- winner
			// 4, 1
			4.26, 1.65, 1, // 12
			4.61, 1.3, 1,  // 13 <--- winner
			4.00, 1.0, 1,  // 14
			// 3, 1
			3.21, 1.25, 1, // 15 <--- winner
			// 2, 1
			2.21, 1.65, 1, // 16 <--- winner
			2.36, 1.15, 1, // 17
			2.76, 1.75, 1, // 18
			// 2, 0
			2.30, 0.35, 1, // 19 <--- winner
			// 5, 0
			5.71, 0.8, 1,  // 20
			5.11, 0.65, 1, // 21
			5.11, 0.4, 1,  // 22
			5.51, 0.4, 1,  // 23 <--- winner
			5.91, 0.25, 1, // 24
			// 5, 3
			5.16, 3.25, 1, // 25
			5.46, 3.65, 1, // 26 <--- winner
			5.71, 3.3, 1,  // 27
			// 5, 2
			5.46, 2.75, 1, // 28
			5.31, 2.45, 1, // 29 <--- winner
			5.51, 2.2, 1,  // 30
			// 4, 2
			4.41, 2.65, 1,  // 31 <--- winner
			// 4, 0
			4.62, 0.3, 1   // 32 <--- winner
	};
	//@formatter:on
	o3c::Tensor nodes(node_data, {33, 3}, o3c::Dtype::Float32, device);
	ngeom::HierarchicalGraphWarpField hgwf(
			nodes, 0.25, false, 4, 0, 3, 4,
			[](int i_layer, float node_coverage) {
				return powf(2.f, static_cast<float>(i_layer)) * node_coverage;
			}
	);

	//@formatter:off
	o3c::Tensor ground_truth_layer_1 = nnrt::core::functional::SortTensorAlongLastDimension(o3c::Tensor(std::vector<int>{
			//2-4, 0-2
			16, // 2.21, 1.65, 1.0,
			19, // 2.3 , 0.35, 1.0,
			15, // 3.21, 1.25, 1.0, <--- definite winner
			//2-4, 2-4
			4,  // 2.31, 2.75, 1.0, <--- definite winner
			1,  // 2.32, 3.7 , 1.0,
			3,  // 3.36, 3.7 , 1.0,
			10, // 3.56, 2.3 , 1.0,
			//4-6, 2-4
			11, // 4.21, 3.7 , 1.0,
			31, // 4.41, 2.65, 1.0, <--- definite winner, ~3.441 distance sum to other nodes in group
			29, // 5.31, 2.45, 1.0,
			26, // 5.46, 3.65, 1.0,
			//4-6, 0-2
			13, // 4.61, 1.3, 1.0,
			23, // 5.51, 0.4, 1.0,
			32, // 4.62, 0.3, 1.0   <--- definite winner
	}, {14}, o3c::Int32, device));
	//@formatter:on


	o3c::Tensor ground_truth_layer_2 = nnrt::core::functional::SortTensorAlongLastDimension(o3c::Tensor(std::vector<int>{
			4, // 2.31, 2.75, 1.0,
			15,// 3.21, 1.25, 1.0,
			31,// 4.41, 2.65, 1.0,
			32,//  4.62, 0.3, 1.0
	}, {4}, o3c::Int32, device));


	auto layer_1_nodes = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetRegularizationLevel(1).node_indices);
	auto layer_2_nodes = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetRegularizationLevel(2).node_indices);
	REQUIRE(layer_1_nodes.AllEqual(ground_truth_layer_1));
	REQUIRE(layer_2_nodes.AllEqual(ground_truth_layer_2));


	open3d::core::Tensor edges = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetEdges());
	auto& edge_weights = hgwf.GetEdgeWeights();
	auto& virtual_node_indices = hgwf.GetVirtualNodeIndices();

	o3c::Tensor edges_gt = nnrt::core::functional::SortTensorAlongLastDimension(
			o3c::Tensor(std::vector<int>{
					1, 4, 3, 16,
					1, 10, 4, 3,
					1, 10, 4, 3,
					1, 10, 3, 11,
					1, 10, 4, 16,
					10, 4, 15, 16,
					1, 10, 4, 16,
					10, 4, 3, 31,
					10, 3, 11, 31,
					10, 4, 15, 16,
					10, 4, 15, 31,
					26, 3, 11, 31,
					10, 13, 15, 31,
					29, 23, 13, 32,
					10, 13, 15, 32,
					10, 19, 15, 16,
					4, 19, 15, 16,
					4, 19, 15, 16,
					10, 4, 15, 16,
					19, 15, 16, 32,
					29, 23, 13, 32,
					29, 23, 13, 32,
					29, 23, 13, 32,
					29, 23, 13, 32,
					29, 23, 13, 32,
					29, 26, 11, 31,
					29, 26, 11, 31,
					29, 26, 11, 31,
					29, 26, 11, 31,
					29, 26, 13, 31,
					29, 26, 13, 31,
					10, 29, 11, 31,
					23, 13, 15, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32,
					4, 15, 31, 32
			}, {47, 4}, o3c::Int32, device)
	);

	o3c::Tensor edge_weights_gt(std::vector<float>{
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
	}, {47}, o3c::Float32, device);

	o3c::Tensor virtual_node_indices_gt(std::vector<int>{
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 26, 1, 10, 16,
			29, 31, 32, 19, 23, 13, 15, 3,
			4, 11, 31, 32, 4, 15
	}, {51}, o3c::Int32, device);

	REQUIRE(edges.AllEqual(edges_gt));
	REQUIRE(edge_weights.AllEqual(edge_weights_gt));
	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 0, 33)).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0, 0, 33))));
	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 33, 33 + 14)).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0, 33, 33 + 14))));
	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 33 + 14, 33 + 14 + 4)).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0, 33 + 14, 33 + 14 + 4))));
}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}