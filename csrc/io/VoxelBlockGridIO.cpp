//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/8/22.
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
#include "VoxelBlockGridIO.h"
#include "geometry/VoxelBlockGrid.h"
#include "io/TensorIO.h"
#include "io/DeviceIO.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

namespace nnrt::io {
std::ostream& operator<<(std::ostream& ostream, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid) {
	// write header (voxel size, block resolution)
	ostream.write(reinterpret_cast<const char*>(&voxel_block_grid.voxel_size_), sizeof(float));
	ostream.write(reinterpret_cast<const char*>(&voxel_block_grid.block_resolution_), sizeof(int64_t));
	// write name attribute count and map
	auto attribute_count = static_cast<int32_t>(voxel_block_grid.name_attr_map_.size());
	ostream.write(reinterpret_cast<const char*>(&attribute_count), sizeof(int32_t));
	for (auto& key_value_pair: voxel_block_grid.name_attr_map_) {
		ostream << key_value_pair.first;
		ostream.write(reinterpret_cast<const char*>(&key_value_pair.second), sizeof(int));
	}
	// write hashmap properties and contents
	// device & capacity
	ostream << voxel_block_grid.block_hashmap_->GetDevice();
	int64_t hash_map_capacity = voxel_block_grid.block_hashmap_->GetCapacity();
	ostream.write(reinterpret_cast<const char*>(&hash_map_capacity), sizeof(int32_t));
	// keys & values
	o3c::Tensor keys = voxel_block_grid.block_hashmap_->GetKeyTensor();
	ostream << keys;
	std::vector<o3c::Tensor> value_tensors = voxel_block_grid.block_hashmap_->GetValueTensors();
	for (auto& value_tensor: value_tensors) {
		ostream << value_tensor;
	}
	return ostream;
}

std::istream& operator>>(std::istream& istream, geometry::VoxelBlockGrid& voxel_block_grid) {
	// read header (voxel size and block resolution)
	istream.read(reinterpret_cast<char*>(&voxel_block_grid.voxel_size_), sizeof(float));
	istream.read(reinterpret_cast<char*>(&voxel_block_grid.block_resolution_), sizeof(int64_t));
	// read attribute count & map
	int32_t attribute_count;
	istream.read(reinterpret_cast<char*>(&attribute_count), sizeof(int32_t));
	voxel_block_grid.name_attr_map_.clear();
	for (int32_t i_attribute = 0; i_attribute < attribute_count; i_attribute++) {
		std::string attribute_name;
		istream >> attribute_name;
		int attribute_index;
		istream.read(reinterpret_cast<char*>(&attribute_index), sizeof(int));
		voxel_block_grid.name_attr_map_[attribute_name] = attribute_index;
	}
	// read hashmap properties and contents
	// device & capacity
	o3c::Device hash_map_device;
	istream >> hash_map_device;
	int64_t hash_map_capacity;
	istream.read(reinterpret_cast<char*>(&hash_map_capacity), sizeof(int64_t));
	// keys & values
	o3c::Tensor keys;
	istream >> keys;
	std::vector<o3c::Tensor> attribute_value_tensors;
	std::vector<o3c::Dtype> attribute_dtypes;
	std::vector<o3c::SizeVector> attribute_element_shapes;
	for (int32_t i_attribute = 0; i_attribute < attribute_count; i_attribute++) {
		o3c::Tensor value_tensor;
		istream >> value_tensor;
		attribute_value_tensors.push_back(value_tensor);
		attribute_dtypes.push_back(value_tensor.GetDtype());
		o3c::SizeVector attribute_element_shape(value_tensor.GetShape());
		attribute_element_shape.erase(attribute_element_shape.begin());
		attribute_element_shapes.push_back(attribute_element_shape);
	}
	voxel_block_grid.block_hashmap_ = std::make_shared<o3c::HashMap>(
			hash_map_capacity, o3c::Int32, o3c::SizeVector{3}, attribute_dtypes,
			attribute_element_shapes, hash_map_device, o3c::HashBackendType::Default
	);
	voxel_block_grid.block_hashmap_->Insert(keys, attribute_value_tensors);
}

} // nnrt::io