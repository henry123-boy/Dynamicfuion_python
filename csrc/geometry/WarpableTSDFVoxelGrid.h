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
#include <geometry/DualQuaternion.h>

namespace nnrt {
namespace geometry {

class WarpableTSDFVoxelGrid : public open3d::t::geometry::TSDFVoxelGrid {

public:
	using TSDFVoxelGrid::TSDFVoxelGrid;

/// Extract all indexed voxel centers.
	open3d::core::Tensor ExtractVoxelCenters();

	/// Extract all TSDF values in the same order as the voxel centers in the output
	/// of the ExtractVoxelCenters function
	open3d::core::Tensor ExtractTSDFValuesAndWeights();

	/// Extract all SDF values in the specified spatial extent
	/// All undefined SDF values will be kept as -2.0
	open3d::core::Tensor ExtractValuesInExtent(int min_voxel_x, int min_voxel_y, int min_voxel_z, int max_voxel_x, int max_voxel_y, int max_voxel_z);


	open3d::core::Tensor IntegrateWarpedEuclideanDQ(const open3d::t::geometry::Image& depth,
	                                                const open3d::core::Tensor& depth_normals,
	                                                const open3d::core::Tensor& intrinsics,
	                                                const open3d::core::Tensor& extrinsics,
	                                                const open3d::core::Tensor& warp_graph_nodes,
	                                                const open3d::core::Tensor& node_dual_quaternion_transformations,
	                                                float node_coverage,
	                                                int anchor_count = 4,
	                                                int minimum_valid_anchor_count = 3,
	                                                float depth_scale = 1000.0f,
	                                                float depth_max = 3.0f);

	open3d::core::Tensor IntegrateWarpedEuclideanDQ(const open3d::t::geometry::Image& depth,
	                                                const open3d::t::geometry::Image& color,
	                                                const open3d::core::Tensor& depth_normals,
	                                                const open3d::core::Tensor& intrinsics,
	                                                const open3d::core::Tensor& extrinsics,
	                                                const open3d::core::Tensor& warp_graph_nodes,
			// TODO: provide a more intuitive handling of quaternions, e.g.
			//  python (numpy.ndarray w/ dtype=DualQuaternion)<--> CPU code (MemoryBlock<DualQuaternion>) <--> CUDA code (MemoryBlock<DualQuaternion>)
			//  a good CPU<-->CUDA implementation of MemoryBlock<CUDA-compatible-type>
			                             const open3d::core::Tensor& node_dual_quaternion_transformations,
			                                        float node_coverage,
	                                                int anchor_count = 4,
	                                                int minimum_valid_anchor_count = 3,
	                                                float depth_scale = 1000.0f,
	                                                float depth_max = 3.0f);

	open3d::core::Tensor IntegrateWarpedEuclideanMat(const open3d::t::geometry::Image& depth,
	                                                 const open3d::t::geometry::Image& color,
	                                                 const open3d::core::Tensor& depth_normals,
	                                                 const open3d::core::Tensor& intrinsics,
	                                                 const open3d::core::Tensor& extrinsics,
	                                                 const open3d::core::Tensor& warp_graph_nodes,
	                                                 const open3d::core::Tensor& node_rotations,
	                                                 const open3d::core::Tensor& node_translations,
	                                                 float node_coverage,
	                                                 int anchor_count = 4,
	                                                 int minimum_valid_anchor_count = 3,
	                                                 float depth_scale = 1000.0f,
	                                                 float depth_max = 3.0f);

	open3d::core::Tensor IntegrateWarpedEuclideanMat(const open3d::t::geometry::Image& depth,
	                                                 const open3d::core::Tensor& depth_normals,
	                                                 const open3d::core::Tensor& intrinsics,
	                                                 const open3d::core::Tensor& extrinsics,
	                                                 const open3d::core::Tensor& warp_graph_nodes,
	                                                 const open3d::core::Tensor& node_rotations,
	                                                 const open3d::core::Tensor& node_translations,
	                                                 float node_coverage,
	                                                 int anchor_count = 4,
	                                                 int minimum_valid_anchor_count = 3,
	                                                 float depth_scale = 1000.0f,
	                                                 float depth_max = 3.0f);

    int64_t ActivateSleeveBlocks();

protected:
	open3d::core::Tensor BufferCoordinatesOfInactiveNeighborBlocks(
			const open3d::core::Tensor &active_block_addresses);

	using TSDFVoxelGrid::voxel_size_;
	using TSDFVoxelGrid::sdf_trunc_;
	using TSDFVoxelGrid::block_resolution_;
	using TSDFVoxelGrid::block_count_;
	using TSDFVoxelGrid::device_;
	using TSDFVoxelGrid::attr_dtype_map_;
};

}// namespace nnrt
}// namespace geometry





