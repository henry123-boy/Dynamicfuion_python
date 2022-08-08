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

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

namespace nnrt::io {
std::ostream& operator<<(std::ostream& ostream, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid) {
	// write header
	ostream.write(reinterpret_cast<const char*>(&voxel_block_grid.voxel_size_), sizeof(float));
	ostream.write(reinterpret_cast<const char*>(&voxel_block_grid.block_resolution_), sizeof(int64_t));
	auto attribute_count = static_cast<int32_t>(voxel_block_grid.name_attr_map_.size());
	ostream.write(reinterpret_cast<const char*>(&attribute_count), sizeof(int32_t));
	// write name attribute map
	for (auto& key_value_pair: voxel_block_grid.name_attr_map_) {
		ostream << key_value_pair.first;
		ostream.write(reinterpret_cast<const char*>(&key_value_pair.second), sizeof(int));
	}
	// write voxel block hash map capacity & contents
	// TODO device & size
	o3c::Tensor keys = voxel_block_grid.block_hashmap_->GetKeyTensor();
	ostream << keys;
	std::vector<o3c::Tensor> value_tensors = voxel_block_grid.block_hashmap_->GetValueTensors();
	for (auto& value_tensor: value_tensors) {
		ostream << value_tensor;
	}

	return ostream;
}

std::istream& operator>>(std::istream& istream, geometry::VoxelBlockGrid& voxel_block_grid) {
	istream.read(reinterpret_cast<char*>(&voxel_block_grid.voxel_size_), sizeof(float));
	istream.read(reinterpret_cast<char*>(&voxel_block_grid.block_resolution_), sizeof(int64_t));
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
	o3c::Tensor keys;
	istream >> keys;
	std::vector<o3c::Tensor> value_tensors;
	for(int32_t i_attribute = 0; i_attribute < attribute_count; i_attribute++) {
		o3c::Tensor value_tensor;
		istream >> value_tensor;
		value_tensors.push_back(value_tensor);
	}


}

} // nnrt::io