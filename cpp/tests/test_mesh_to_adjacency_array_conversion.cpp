//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/1/23.
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
// third-party includes
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/io/TriangleMeshIO.h>

// local includes
#include "test_main.hpp"
#include "tests/test_utils/test_utils.hpp"

// code being tested
#include "geometry/functional/TopologicalConversions.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

void TestTriangleMeshToAdjacencyArrayConversion(const o3c::Device& device){
	o3tg::TriangleMesh sphere = o3tg::TriangleMesh::CreateSphere(1.0, 8, o3c::Float32, o3c::Int64, device);
	o3c::Tensor adjacency_array = nnrt::geometry::functional::MeshToAdjacencyArray(sphere, 16);
	//TODO: move test data to data pack
	//test data generated via single-threaded python code
	o3c::Tensor adjacency_array_gt = o3c::Tensor::Load(test::static_array_test_data_directory.ToString() + "/sphere16_adj.npy").To(device);
	REQUIRE(adjacency_array.AllEqual(adjacency_array_gt));
}

TEST_CASE("Test Mest to Adjacency Array - CPU") {
	auto device = o3c::Device("CPU:0");
	TestTriangleMeshToAdjacencyArrayConversion(device);
}

TEST_CASE("Test Mest to Adjacency Array - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestTriangleMeshToAdjacencyArrayConversion(device);
}
