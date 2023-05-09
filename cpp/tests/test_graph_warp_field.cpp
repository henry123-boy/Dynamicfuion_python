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
	std::vector<float> node_data{
			// 2, 3
			2.11, 3.2, 1,
			2.32, 3.7, 1,
			2.66, 3.35, 1,
			// 3, 3
			3.36, 3.7, 1,
			// 2, 2
			2.31, 2.75, 1,
			2.41, 2.2, 1,
			2.71, 2.7, 1,
			// 3, 2
			3.31, 2.8, 1,
			3.71, 2.8, 1,
			3.16, 2.3, 1,
			3.56, 2.3, 1,
			// 4, 3
			4.21, 3.7, 1,
			// 4, 1
			4.26, 1.65, 1,
			4.61, 1.3, 1,
			4.00, 1.0, 1,
			// 3, 1
			3.21, 1.25, 1,
			// 2, 1
			2.21, 1.65, 1,
			2.36, 1.15, 1,
			2.76, 1.75, 1,
			// 2, 0
			2.30, 0.35, 1,
			// 5, 0
			5.71, 0.8, 1,
			5.11, 0.65, 1,
			5.11, 0.4, 1,
			5.51, 0.4, 1,
			5.91, 0.25, 1,
			// 5, 3
			5.16, 3.25, 1,
			5.46, 3.65, 1,
			5.71, 3.3, 1,
			// 5, 2
			5.46, 2.75, 1,
			5.31, 2.45, 1,
			5.51, 2.2, 1,
			// 4, 2
			4.41, 2.65, 1
	};
	o3c::Tensor nodes(node_data, {32, 3}, o3c::Dtype::Float32, device);
	ngeom::HierarchicalGraphWarpField hgwf(
			nodes, 0.25, false, 4, 0, 3, 4,
			[](int i_layer, float node_coverage) {
				return powf(2.f, static_cast<float>(i_layer)) * node_coverage;
			}
	);

	o3c::Tensor ground_truth_layer_1 = nnrt::core::functional::SortTensorByColumn(o3c::Tensor(std::vector<float>{
			//2-4, 0-2
			2.21, 1.65, 1.0,
			2.3, 0.35, 1.0,
			3.21, 1.25, 1.0, // definite winner
			//2-4, 2-4
			2.31, 2.75, 1.0, // definite winner
			2.32, 3.7, 1.0,
			3.36, 3.7, 1.0,
			3.56, 2.3, 1.0,
			//4-6, 2-4
			4.21, 3.7, 1.0,
			4.41, 2.65, 1.0, // definite winner, ~3.441 distance sum to other nodes in group
			5.31, 2.45, 1.0,
			5.46, 3.65, 1.0,
			//4-6, 0-2 (2 nodes, ambiguous case)
			4.61, 1.3, 1.0,
			5.51, 0.4, 1.0
	}, {13, 3}, o3c::Float32, device), 0);


	o3c::Tensor ground_truth_layer_2_v1 = nnrt::core::functional::SortTensorByColumn(o3c::Tensor(std::vector<float>{
			2.31, 2.75, 1.0,
			3.21, 1.25, 1.0,
			4.41, 2.65, 1.0,
			5.51, 0.4, 1.0
	}, {4, 3}, o3c::Float32, device), 0);

	o3c::Tensor ground_truth_layer_2_v2 = nnrt::core::functional::SortTensorByColumn(o3c::Tensor(std::vector<float>{
			2.31, 2.75, 1.0,
			3.21, 1.25, 1.0,
			4.41, 2.65, 1.0,
			4.61, 1.3, 1.0
	}, {4, 3}, o3c::Float32, device), 0);


	auto layer_1_nodes = nnrt::core::functional::SortTensorByColumn(hgwf.GetRegularizationLevel(1).nodes, 0);
	auto layer_2_nodes = nnrt::core::functional::SortTensorByColumn(hgwf.GetRegularizationLevel(2).nodes, 0);
	REQUIRE(layer_1_nodes.AllClose(ground_truth_layer_1));
	REQUIRE((layer_2_nodes.AllClose(ground_truth_layer_2_v1) || layer_2_nodes.AllClose(ground_truth_layer_2_v2)));

}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}