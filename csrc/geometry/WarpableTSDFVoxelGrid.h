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

#include <open3d/t/geometry/TSDFVoxelGrid.h>
#include <geometry/DualQuaternion.h>

namespace nnrt {
namespace geometry {

class WarpableTSDFVoxelGrid : public open3d::t::geometry::TSDFVoxelGrid {

public:
	using TSDFVoxelGrid::TSDFVoxelGrid;
	WarpableTSDFVoxelGrid(std::unordered_map<std::string, open3d::core::Dtype> attr_dtype_map =
			{{"tsdf",   open3d::core::Dtype::Float32},
			 {"weight", open3d::core::Dtype::UInt16},
			 {"color",  open3d::core::Dtype::UInt16}},
	                      float voxel_size = 3.0 / 512.0, /* in meter */
	                      float sdf_trunc = 0.04,         /*  in meter  */
	                      int64_t block_resolution = 16, /*  block Tensor resolution  */
	                      int64_t block_count = 1000,
	                      int64_t anchor_count = 4,
	                      const open3d::core::Device& device = open3d::core::Device("CPU:0"),
	                      const open3d::core::HashmapBackend& backend =
	                      open3d::core::HashmapBackend::Default);
/// Extract all indexed voxel centers.
	open3d::core::Tensor ExtractVoxelCenters();

	/// Extract all TSDF values in the same order as the voxel centers in the output
	/// of the ExtractVoxelCenters function
	open3d::core::Tensor ExtractTSDFValuesAndWeights();

	/// Extract all SDF values in the specified spatial extent
	/// All undefined SDF values will be kept as -2.0
	open3d::core::Tensor ExtractValuesInExtent(int min_x, int min_y, int min_z, int max_x, int max_y, int max_z);

	//__DEBUG
	open3d::core::Tensor IntegrateWarped(const open3d::t::geometry::Image& depth,
	                                     const open3d::core::Tensor& depth_normals,
	                                     const open3d::core::Tensor& intrinsics,
	                                     const open3d::core::Tensor& extrinsics,
	                                     const open3d::core::Tensor& warp_graph_nodes,
	                                     const open3d::core::Tensor& node_dual_quaternion_transformations,
	                                     float node_coverage,
	                                     float depth_scale,
	                                     float depth_max);


	open3d::core::Tensor IntegrateWarped(const open3d::t::geometry::Image& depth,
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
			                             float depth_scale,
			                             float depth_max);

protected:
	int64_t anchor_node_count_;
	using TSDFVoxelGrid::voxel_size_;
	using TSDFVoxelGrid::sdf_trunc_;
	using TSDFVoxelGrid::block_resolution_;
	using TSDFVoxelGrid::block_count_;
	using TSDFVoxelGrid::device_;
	using TSDFVoxelGrid::attr_dtype_map_;
};

}// namespace nnrt
}// namespace geometry





