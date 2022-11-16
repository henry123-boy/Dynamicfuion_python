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
#include <open3d/t/geometry/Image.h>

#include "geometry/VoxelBlockGrid.h"
#include "geometry/functional/AnchorComputationMethod.h"
#include "geometry/GraphWarpField.h"
#include "io/VoxelBlockGridIO.h"

namespace nnrt::geometry {

class NonRigidSurfaceVoxelBlockGrid : public VoxelBlockGrid {

public:
	using VoxelBlockGrid::VoxelBlockGrid;

	open3d::core::Tensor FindBlocksIntersectingTruncationRegion(
			const open3d::t::geometry::Image& depth, const GraphWarpField& warp_field, const open3d::core::Tensor& intrinsics,
			const open3d::core::Tensor& extrinsics, float depth_scale = 1000.f, float depth_max = 3.f, float truncation_voxel_multiplier = 8.f
	) const;

	open3d::core::Tensor IntegrateNonRigid(
			const open3d::core::Tensor& block_coords, const GraphWarpField& warp_field,
			const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color, const open3d::core::Tensor& depth_normals,
			const open3d::core::Tensor& depth_intrinsics, const open3d::core::Tensor& color_intrinsics, const open3d::core::Tensor& extrinsics,
			float depth_scale = 1000.f, float depth_max = 3.f, float truncation_voxel_multiplier = 8.f
	);

	int64_t ActivateSleeveBlocks();

	open3d::core::Tensor ExtractVoxelValuesAndCoordinates();
	open3d::core::Tensor ExtractVoxelValuesAt(const open3d::core::Tensor& query_voxel_coordinates);
	open3d::core::Tensor ExtractVoxelBlockCoordinates();

	friend std::ostream& nnrt::io::operator<<(std::ostream& ostream, const NonRigidSurfaceVoxelBlockGrid& voxel_block_grid);
	friend std::istream& nnrt::io::operator>>(std::istream& istream, NonRigidSurfaceVoxelBlockGrid& voxel_block_grid);

protected:
	open3d::core::Tensor BufferCoordinatesOfInactiveNeighborBlocks(const open3d::core::Tensor& active_block_addresses) const;

	/// Get minimum extent (min & max corner) coordinates of bounding boxes for projected block coordinates
	open3d::core::Tensor GetBoundingBoxesOfWarpedBlocks(const open3d::core::Tensor& block_keys, const GraphWarpField& warp_field,
	                                                    const open3d::core::Tensor& extrinsics) const;
	open3d::core::Tensor GetAxisAlignedBoxesIntersectingSurfaceMask(const open3d::core::Tensor& boxes, const open3d::t::geometry::Image& depth,
	                                                                const open3d::core::Tensor& intrinsics, float depth_scale,
	                                                                float depth_max, int downsampling_factor = 4,
	                                                                float trunc_voxel_multiplier = 8.0) const;
};

}// namespace geometry