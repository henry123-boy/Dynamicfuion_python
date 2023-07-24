//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/18/23.
//  Copyright (c) 2023 Gregory Kramida
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
// stdlib includes

// third-party includes

// local includes
#include "test_main.hpp"
#include "tests/test_utils/test_utils.hpp"

// code being tested
#include "geometry/functional/WarpAnchorComputation.h"
#include "core/functional/Sorting.h"


namespace o3c = open3d::core;

void TestComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(const o3c::Device& device) {
	o3c::Tensor vertices(std::vector<float>{
			-0.0625, 0.3625, 0.,
			-0.0625, 0.2375, 0.,
			0.0625, 0.3625, 0.,
			0.0625, 0.2375, 0.,
			-0.0625, -0.2375, 0.,
			-0.0625, -0.3625, 0.,
			0.0625, -0.2375, 0.,
			0.0625, -0.3625, 0.
	}, {8, 3}, o3c::Float32, device);

	o3c::Tensor nodes(std::vector<float>{
			0.0, 0.3, 0.0,
			0.0, -0.3, 0.0,
	}, {2, 3}, o3c::Float32, device);

	o3c::Tensor node_weights(std::vector<float>{
			0.36,
			0.36
	}, {2}, o3c::Float32, device);

	o3c::Tensor anchors_gt(std::vector<int32_t>{
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1,
			1, 0, -1, -1
	}, {8, 4}, o3c::Int32, device);

	o3c::Tensor weights_gt(std::vector<float>{
			0.64660899, 0.35339101, 0., 0.,
			0.59768616, 0.40231384, 0., 0.,
			0.64660899, 0.35339101, 0., 0.,
			0.59768616, 0.40231384, 0., 0.,
			0.59768616, 0.40231384, 0., 0.,
			0.64660899, 0.35339101, 0., 0.,
			0.59768616, 0.40231384, 0., 0.,
			0.64660899, 0.35339101, 0., 0.
	}, {8, 4}, o3c::Float32, device);

	o3c::Tensor anchors, weights;

	nnrt::geometry::functional::ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(anchors, weights, vertices, nodes, node_weights, 4, 0);

	weights = nnrt::core::functional::SortTensorAlongLastDimension(weights, true, nnrt::core::functional::SortOrder::DESC);
	anchors = nnrt::core::functional::SortTensorAlongLastDimension(anchors, true, nnrt::core::functional::SortOrder::DESC);

	REQUIRE(weights.AllClose(weights_gt, 1e-3, 1e-6));
	REQUIRE(anchors.AllEqual(anchors_gt));
}


TEST_CASE("Test Compute Anchors - Euclidean - VariableNodeWeight - CPU") {
	auto device = o3c::Device("CPU:0");
	TestComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(device);
}

TEST_CASE("Test Compute Anchors - Euclidean - VariableNodeWeight - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(device);
}

