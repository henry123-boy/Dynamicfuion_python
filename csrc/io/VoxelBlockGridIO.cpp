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

namespace nnrt::io {
std::ostream& operator<<(std::ostream& ostream, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid) {
	// write header
	float voxel_size = grid.GetVoxelSize();
	int64_t block_resolution = grid.GetBlockResolution();
	int64_t block_count = grid.GetBlockCount();
	ostream.write(reinterpret_cast<const char*>(&voxel_block_grid.voxel_size_), sizeof(float));
	ostream.write(reinterpret_cast<const char*>(&block_resolution), sizeof(int64_t));
	ostream.write(reinterpret_cast<const char*>(&block_count), sizeof(int64_t));
	// write device type members (since no serialization is provided for that)
	int device_id = grid.GetDevice().GetID();
	ostream.write(reinterpret_cast<const char*>(&device_id), sizeof(int));
	o3c::Device::DeviceType device_type = grid.GetDevice().GetType();
	ostream.write(reinterpret_cast<const char*>(&device_type), sizeof(o3c::Device::DeviceType));
	//TODO
	throw std::runtime_error("Not implemented.");

	return ostream;
}

std::istream& operator>>(std::istream& istream, geometry::VoxelBlockGrid& voxel_block_grid) {
	return istream;
}

} // nnrt::io