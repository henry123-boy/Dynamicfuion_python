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
#pragma once

#include <utility>

#include <open3d/t/geometry/TSDFVoxelGrid.h>
#include "geometry/AnchorComputationMethod.h"
#include "geometry/GraphWarpField.h"

namespace nnrt::geometry {

class WarpableTSDFVoxelGrid : public open3d::t::geometry::TSDFVoxelGrid {

public:
	using TSDFVoxelGrid::TSDFVoxelGrid;

	friend std::ostream& operator<<(std::ostream& out, const WarpableTSDFVoxelGrid& grid);


/// Extract all indexed voxel centers.
	open3d::core::Tensor ExtractVoxelCenters();

	/// Extract all TSDF residuals in the same order as the voxel centers in the output
	/// of the ExtractVoxelCenters function
	open3d::core::Tensor ExtractTSDFValuesAndWeights();

	/// Extract all SDF residuals in the specified spatial extent
	/// All undefined SDF residuals will be kept as -2.0
	open3d::core::Tensor ExtractValuesInExtent(int min_voxel_x, int min_voxel_y, int min_voxel_z, int max_voxel_x, int max_voxel_y, int max_voxel_z);

	open3d::core::Tensor
	IntegrateWarped(const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color, const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& intrinsics,
	                const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
	                float depth_scale = 1000.0f, float depth_max = 3.0f);

	int64_t ActivateSleeveBlocks();

protected:
	open3d::core::Tensor BufferCoordinatesOfInactiveNeighborBlocks(
			const open3d::core::Tensor& active_block_addresses);
};

}// namespace geometry





