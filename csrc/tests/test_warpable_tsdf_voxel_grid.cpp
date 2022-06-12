//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/9/22.
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
#include <fmt/ranges.h>

#include "test_main.hpp"

#include <geometry/WarpableTSDFVoxelGrid.h>
#include <geometry/kernel/WarpableTSDFVoxelGrid.h>
#include <geometry/GraphWarpField.h>


using namespace nnrt;
namespace o3c = open3d::core;


void TestWarpableTSDFVoxelGridConstructor(const o3c::Device& device) {
	// simply ensure construction works without bugs
	nnrt::geometry::WarpableTSDFVoxelGrid grid(
			{
					{"tsdf",   o3c::Float32},
					{"weight", o3c::UInt16},
					{"color",  o3c::UInt16}
			},
			0.005f,
			0.025f,
			16,
			1000,
			device
	);
}

TEST_CASE("Test Warpable TSDF Voxel Grid Constructor CPU") {
	auto device = o3c::Device("CPU:0");
	TestWarpableTSDFVoxelGridConstructor(device);
}

TEST_CASE("Test Warpable TSDF Voxel Grid Constructor CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestWarpableTSDFVoxelGridConstructor(device);
}

void TestWarpableTSDFVoxelGrid_GetBoundingBoxesOfWarpedBlocks(const o3c::Device& device) {
	std::vector<int32_t> block_key_data{
			0, 0, 0,
			0, 1, 0,
			0, 0, 1,
			0, 1, 1
	};
	o3c::Tensor block_keys{block_key_data, {4, 3}, o3c::Dtype::Int32, device};

	float voxel_size = 0.1;
	int64_t block_resolution = 10;

	std::vector<float> node_data{
			0.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0,
			0.0, 1.0, 1.0,
	};
	o3c::Tensor nodes(node_data, {4, 3}, o3c::Dtype::Float32, device);
	std::vector<int> edge_data{
			1, 2, -1, -1,
			0, 1, -1, -1,
			0, 2, -1, -1,
			1, 2, -1, -1
	};
	o3c::Tensor edges(edge_data, {4, 4}, o3c::Dtype::Int32, device);
	o3c::Tensor edge_weights = o3c::Tensor::Ones({4, 4}, o3c::Dtype::Float32, device);
	o3c::Tensor clusters = o3c::Tensor::Ones({4}, o3c::Dtype::Float32, device);


	geometry::GraphWarpField field(nodes, edges, edge_weights, clusters, 1.0, false, 4);
	std::vector<float> translation_data{1.0, 0.0, 0.0};
	field.translations.SetItem(o3c::TensorKey::Index(0), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(1), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(2), o3c::Tensor(translation_data, {3}, o3c::Float32));
	field.translations.SetItem(o3c::TensorKey::Index(3), o3c::Tensor(translation_data, {3}, o3c::Float32));


	o3c::Tensor bounding_boxes;
	geometry::kernel::tsdf::GetBoundingBoxesOfWarpedBlocks(bounding_boxes, block_keys, field, voxel_size, block_resolution,
	                                                       o3c::Tensor::Eye(4,o3c::Float64, o3c::Device("CPU:0")));


	auto out = bounding_boxes.ToFlatVector<float>();
	int i = 0;
	for(auto& f : out){
		if(i % 6 == 0){
			std::cout << std::endl;
		}
		std::cout << f << ", ";
		i++;
	}
	std::cout << std::endl;
	std::cout << out[18] << std::endl;

}

TEST_CASE("Test Warpable TSDF Voxel Grid GetBoundingBoxesOfWarpedBlocks CPU") {
	auto device = o3c::Device("CPU:0");
	TestWarpableTSDFVoxelGrid_GetBoundingBoxesOfWarpedBlocks(device);
}

TEST_CASE("Test Warpable TSDF Voxel Grid GetBoundingBoxesOfWarpedBlocks CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestWarpableTSDFVoxelGrid_GetBoundingBoxesOfWarpedBlocks(device);
}


