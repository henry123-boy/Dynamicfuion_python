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
// test framework
#include "test_main.hpp"

// 3rd party
#include <open3d/core/Tensor.h>

// code being tested
#include "geometry/GraphWarpField.h"

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
	o3c::Tensor edge_weights(edge_weight_data, {3,4}, o3c::Dtype::Float32, device);
	std::vector<int> clusters_data{0,0,0};
	o3c::Tensor clusters(clusters_data, {3}, o3c::Dtype::Int32, device);

	ngeom::GraphWarpField gwf(nodes, edges, edge_weights, clusters);
	REQUIRE(gwf.nodes.GetShape(0) == 3);
	REQUIRE(gwf.nodes.GetShape(1) == 3);
	REQUIRE(gwf.nodes.ToFlatVector<float>() == nodes.ToFlatVector<float>());
	REQUIRE(gwf.edges.ToFlatVector<int>() == edges.ToFlatVector<int>());
	REQUIRE(gwf.edge_weights.ToFlatVector<float>() == edge_weights.ToFlatVector<float>());
	REQUIRE(gwf.clusters.ToFlatVector<int>() == clusters.ToFlatVector<int>());

}

TEST_CASE("Test Graph Warp Field Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestGraphWarpFieldConstructor(device);
}

TEST_CASE("Test Graph Warp Field Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestGraphWarpFieldConstructor(device);
}