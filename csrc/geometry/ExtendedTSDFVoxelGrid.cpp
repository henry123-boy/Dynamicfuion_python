//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
//  Copyright (c) 2021 Gregory Kramida
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
#include "ExtendedTSDFVoxelGrid.h"

#include <open3d/t/geometry/kernel/TSDFVoxelGrid.h>
#include "geometry/kernel/ExtendedTSDFVoxelGrid.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt {
namespace geometry {

core::Tensor ExtendedTSDFVoxelGrid::ExtractVoxelCenters() {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_addrs;
	block_hashmap_->GetActiveIndices(active_addrs);
	core::Tensor active_nb_addrs, active_nb_masks;
	std::tie(active_nb_addrs, active_nb_masks) =
			BufferRadiusNeighbors(active_addrs);

	core::Tensor voxel_centers;
	kernel::tsdf::ExtractVoxelCenters(
			active_addrs.To(core::Dtype::Int64),
			active_nb_addrs.To(core::Dtype::Int64), active_nb_masks,
			block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
			voxel_centers, block_resolution_, voxel_size_);

	return voxel_centers;
}

open3d::core::Tensor ExtendedTSDFVoxelGrid::ExtractValuesInExtent(int min_x, int min_y, int min_z, int max_x, int max_y, int max_z) {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_addrs;
	block_hashmap_->GetActiveIndices(active_addrs);
	core::Tensor active_nb_addrs, active_nb_masks;
	std::tie(active_nb_addrs, active_nb_masks) =
			BufferRadiusNeighbors(active_addrs);


	int64_t min_voxel_x = min_x;
	int64_t min_voxel_y = min_y;
	int64_t min_voxel_z = min_z;
	int64_t max_voxel_x = max_x;
	int64_t max_voxel_y = max_y;
	int64_t max_voxel_z = max_z;


	core::Tensor voxel_values;
	kernel::tsdf::ExtractValuesInExtent(
			min_voxel_x, min_voxel_y, min_voxel_z,
			max_voxel_x, max_voxel_y, max_voxel_z,
			active_addrs.To(core::Dtype::Int64),
			active_nb_addrs.To(core::Dtype::Int64), active_nb_masks,
			block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
			voxel_values, block_resolution_);

	return voxel_values;
}

} // namespace geometry
} // namespace nnrt
