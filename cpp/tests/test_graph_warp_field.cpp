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
#include "geometry/WarpField.h"
#include "geometry/HierarchicalGraphWarpField.h"
#include "geometry/PlanarGraphWarpField.h"
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
	REQUIRE(gwf.node_positions.GetShape(0) == 3);
	REQUIRE(gwf.node_positions.GetShape(1) == 3);


	ngeom::PlanarGraphWarpField pgwf(nodes, edges, edge_weights, clusters);
	REQUIRE(pgwf.node_positions.GetShape(0) == 3);
	REQUIRE(pgwf.node_positions.GetShape(1) == 3);
	REQUIRE(pgwf.node_positions.ToFlatVector<float>() == nodes.ToFlatVector<float>());
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
	o3c::Tensor ground_truth_layer_0 = nnrt::core::functional::SortTensorAlongLastDimension(o3c::Tensor(std::vector<int>{
			// 2, 3
		  0,// 2.11, 3.2, 1,  
       // 1,   2.32, 3.7, 1,   <--- winner
		  2,// 2.66, 3.35, 1, 
		    // 3, 3
       // 3,   3.36, 3.7, 1,   <--- winner
		    // 2, 2
       // 4,   2.31, 2.75, 1,  <--- winner
		  5,// 2.41, 2.2, 1,  
		  6,// 2.71, 2.7, 1,  
		    // 3, 2
		  7,// 3.31, 2.8, 1,  
		  8,// 3.71, 2.8, 1,  
		  9,// 3.16, 2.3, 1,
      // 10,   3.56, 2.3, 1,   <--- winner
		    // 4, 3
      // 11,   4.21, 3.7, 1,   <--- winner
		    // 4, 1
		 12,// 4.26, 1.65, 1, 
      // 13,   4.61, 1.3, 1,   <--- winner
		 14,// 4.00, 1.0, 1,  
		    // 3, 1
      // 15,  3.21, 1.25, 1,   <--- winner
		    // 2, 1
      // 16,   2.21, 1.65, 1,  <--- winner
		 17,// 2.36, 1.15, 1, 
		 18,// 2.76, 1.75, 1, 
		    // 2, 0
      // 19,   2.30, 0.35, 1,  <--- winner
		    // 5, 0
		 20,// 5.71, 0.8, 1,  
		 21,// 5.11, 0.65, 1, 
		 22,// 5.11, 0.4, 1,  
      // 23,   5.51, 0.4, 1,   <--- winner
		 24,// 5.91, 0.25, 1, 
		    // 5, 3
		 25,//5.16, 3.25, 1,  
      // 26,  5.46, 3.65, 1,   <--- winner
		 27,//5.71, 3.3, 1,   
		    // 5, 2
		 28,// 5.46, 2.75, 1, 
      // 29,   5.31, 2.45, 1,  <--- winner
		 30,// 5.51, 2.2, 1,  
		    // 4, 2
      // 31,   4.41, 2.65, 1,  <--- winner
		    // 4, 0
	  // 32 // 4.62, 0.3, 1    <--- winner
	}, {19}, o3c::Int32, device), false);
	//@formatter:on

	//@formatter:off
	o3c::Tensor ground_truth_layer_1 = nnrt::core::functional::SortTensorAlongLastDimension(o3c::Tensor(std::vector<int>{
			//2-4, 0-2
			16, // 2.21, 1.65, 1.0,
			19, // 2.3 , 0.35, 1.0,
			// 15, 3.21, 1.25, 1.0, <--- definite winner, eliminated to be elevated to next layer
			//2-4, 2-4
			// 4,  2.31, 2.75, 1.0, <--- definite winner, -||-
			1,  // 2.32, 3.7 , 1.0,
			3,  // 3.36, 3.7 , 1.0,
			10, // 3.56, 2.3 , 1.0,
			//4-6, 2-4
			11, // 4.21, 3.7 , 1.0,
			// 31,  4.41, 2.65, 1.0, <--- definite winner, eliminated, ~3.441 distance sum to other nodes in group
			29, // 5.31, 2.45, 1.0,
			26, // 5.46, 3.65, 1.0,
			//4-6, 0-2
			13, // 4.61, 1.3, 1.0,
			23, // 5.51, 0.4, 1.0,
			// 32, 4.62, 0.3, 1.0   <--- definite winner, eliminated
	}, {10 /*14 w/o elimination*/}, o3c::Int32, device),false);
	//@formatter:on


	o3c::Tensor ground_truth_layer_2 = nnrt::core::functional::SortTensorAlongLastDimension(o3c::Tensor(std::vector<int>{
			4, // 2.31, 2.75, 1.0,
			15,// 3.21, 1.25, 1.0,
			31,// 4.41, 2.65, 1.0,
			32,//  4.62, 0.3, 1.0
	}, {4}, o3c::Int32, device), false);


	auto layer_0_nodes = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetRegularizationLevel(0).node_indices, false);
	auto layer_1_nodes = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetRegularizationLevel(1).node_indices, false);
	auto layer_2_nodes = nnrt::core::functional::SortTensorAlongLastDimension(hgwf.GetRegularizationLevel(2).node_indices, false);
	REQUIRE(layer_0_nodes.AllEqual(ground_truth_layer_0));
	REQUIRE(layer_1_nodes.AllEqual(ground_truth_layer_1));
	REQUIRE(layer_2_nodes.AllEqual(ground_truth_layer_2));


	const open3d::core::Tensor& edges = hgwf.GetEdges();
	auto& virtual_node_indices = hgwf.GetVirtualNodeIndices();

	o3c::Tensor
			edge_source_virtual_node_indices_gt = o3c::Tensor(
			std::vector<int32_t>({28, 28, 28, 28, 27, 27, 27, 27, 26, 26, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23, 22, 22, 22, 22, 21,
			                      21, 21, 21, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15, 14, 14,
			                      14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6,
			                      6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0}), {116}, o3c::Int32, device
	);

	o3c::Tensor edge_source_virtual_node_indices = edges.Slice(1, 0, 1).Contiguous().Flatten();

	REQUIRE(edge_source_virtual_node_indices_gt.AllEqual(edge_source_virtual_node_indices));

	o3c::Tensor edge_node_indices = nnrt::core::functional::SortTensorByColumns(
			virtual_node_indices.GetItem(o3c::TensorKey::IndexTensor(edges.To(o3c::Int64))),
			{1, 0}
	);


	o3c::Tensor edge_node_indices_gt =
			o3c::Tensor(std::vector<int>{
					0, 1,
					0, 3,
					0, 10,
					0, 16,
					1, 4,
					1, 15,
					1, 31,
					1, 32,
					2, 1,
					2, 3,
					2, 10,
					2, 11,
					3, 4,
					3, 15,
					3, 31,
					3, 32,
					5, 1,
					5, 3,
					5, 10,
					5, 16,
					6, 1,
					6, 3,
					6, 10,
					6, 16,
					7, 1,
					7, 3,
					7, 10,
					7, 11,
					8, 3,
					8, 10,
					8, 11,
					8, 29,
					9, 1,
					9, 3,
					9, 10,
					9, 16,
					10, 4,
					10, 15,
					10, 31,
					10, 32,
					11, 4,
					11, 15,
					11, 31,
					11, 32,
					12, 10,
					12, 13,
					12, 23,
					12, 29,
					13, 4,
					13, 15,
					13, 31,
					13, 32,
					14, 10,
					14, 13,
					14, 19,
					14, 23,
					16, 4,
					16, 15,
					16, 31,
					16, 32,
					17, 10,
					17, 13,
					17, 16,
					17, 19,
					18, 10,
					18, 13,
					18, 16,
					18, 19,
					19, 4,
					19, 15,
					19, 31,
					19, 32,
					20, 10,
					20, 13,
					20, 23,
					20, 29,
					21, 10,
					21, 13,
					21, 23,
					21, 29,
					22, 10,
					22, 13,
					22, 23,
					22, 29,
					23, 4,
					23, 15,
					23, 31,
					23, 32,
					24, 10,
					24, 13,
					24, 23,
					24, 29,
					25, 3,
					25, 11,
					25, 26,
					25, 29,
					26, 4,
					26, 15,
					26, 31,
					26, 32,
					27, 11,
					27, 13,
					27, 26,
					27, 29,
					28, 11,
					28, 13,
					28, 26,
					28, 29,
					29, 4,
					29, 15,
					29, 31,
					29, 32,
					30, 13,
					30, 23,
					30, 26,
					30, 29
			}, {116, 2}, o3c::Int32, device);

	REQUIRE(edge_node_indices_gt.AllEqual(edge_node_indices));

	o3c::Tensor virtual_node_indices_gt(std::vector<int>{
			0,  2,  5,  6,  7,  8,  9, 12, 14, 17, 18, 20, 21, 22, 24, 25, 27, 28, 30,  1, 23, 10,  3, 19, 13, 26, 16, 29, 11, 15,  4, 31, 32
	}, {33}, o3c::Int32, device);

	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 0, 19), false).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0, 0, 19), false)));
	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 19, 19 + 10), false).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0, 19, 19 + 10), false)));
	REQUIRE(nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices.Slice(0, 19 + 10, 19 + 10 + 4), false).AllEqual(
			nnrt::core::functional::SortTensorAlongLastDimension(virtual_node_indices_gt.Slice(0,19 + 10, 19 + 10 + 4), false)));
}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}

TEST_CASE("Test Hierarchical Graph Warp Field Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestHierarchicalGraphWarpFieldConstructor(device);
}