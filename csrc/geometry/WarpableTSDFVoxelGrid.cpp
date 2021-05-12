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
#include "WarpableTSDFVoxelGrid.h"

#include <open3d/t/geometry/kernel/TSDFVoxelGrid.h>

#include <utility>
#include "geometry/kernel/WarpableTSDFVoxelGrid.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt {
namespace geometry {


WarpableTSDFVoxelGrid::WarpableTSDFVoxelGrid(
		std::unordered_map<std::string, core::Dtype> attr_dtype_map,
		float voxel_size,
		float sdf_trunc,
		int64_t block_resolution,
		int64_t block_count,
		int64_t anchor_count,
		const core::Device &device,
		const core::HashmapBackend &backend)
		: TSDFVoxelGrid(std::move(attr_dtype_map), voxel_size, sdf_trunc, block_resolution, block_count, device, backend),
		  anchor_node_count_(anchor_count){
	int64_t total_bytes = block_hashmap_->GetValueBytesize() / (block_resolution_ * block_resolution_ * block_resolution_);
	bool has_anchors = false;
	if (attr_dtype_map_.count("anchor_indices") != 0) {
		has_anchors = true;
		core::Dtype dtype = attr_dtype_map_.at("anchor_indices");
		if (dtype != core::Dtype::UInt16) {
			utility::LogWarning(
					"[WarpableTSDFVoxelGrid] unexpected anchor_indices dtype, please "
					"implement your own Voxel structure in "
					"geometry/kernel/Voxel.h for "
					"dispatching.");
		}
		total_bytes += dtype.ByteSize() * anchor_count;
	}
	bool has_anchor_weights = false;
	if (attr_dtype_map_.count("anchor_weights") != 0) {
		has_anchor_weights = true;
		core::Dtype dtype = attr_dtype_map_.at("anchor_weights");
		if (dtype != core::Dtype::Float32) {
			utility::LogWarning(
					"[WarpableTSDFVoxelGrid] unexpected anchor_weights dtype, please "
					"implement your own Voxel structure in "
					"geometry/kernel/Voxel.h for dispatching.");
		}
		total_bytes += dtype.ByteSize() * anchor_count;
	}
	if ((has_anchors && !has_anchor_weights) || (has_anchor_weights && !has_anchors)){
		utility::LogError(
				"[WarpableTSDFVoxelGrid] unexpected combination of attributes: "
				"expected either both anchor_weights and anchors to be defined "
	            "in the voxel structure or none.");
	}
	block_hashmap_.reset();
	block_hashmap_ = std::make_shared<core::Hashmap>(
			block_count_, core::Dtype::Int32, core::Dtype::UInt8,
			core::SizeVector{3},
			core::SizeVector{block_resolution_, block_resolution_,
			                 block_resolution_, total_bytes},
			device, backend);
}

core::Tensor WarpableTSDFVoxelGrid::ExtractVoxelCenters() {
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


open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractTSDFValuesAndWeights() {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_addrs;
	block_hashmap_->GetActiveIndices(active_addrs);
	core::Tensor active_nb_addrs, active_nb_masks;
	std::tie(active_nb_addrs, active_nb_masks) =
			BufferRadiusNeighbors(active_addrs);

	core::Tensor voxel_values;
	kernel::tsdf::ExtractTSDFValuesAndWeights(
			active_addrs.To(core::Dtype::Int64),
			active_nb_addrs.To(core::Dtype::Int64), active_nb_masks,
			block_hashmap_->GetValueTensor(),
			voxel_values, block_resolution_);

	return voxel_values;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractValuesInExtent(int min_x, int min_y, int min_z, int max_x, int max_y, int max_z) {
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


open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarped(const Image& depth, const Image& color,
                                       const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                       const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                       const core::Tensor& node_dual_quaternion_transformations,
                                       float node_coverage, float depth_scale, float depth_max) {
	throw std::logic_error("ExtendedTSDFVoxelGrid::FuseWarped Function not yet implemented.");
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarped(const Image& depth, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                            const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                            const core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                            float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarped(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_dual_quaternion_transformations,
	                node_coverage, depth_scale, depth_max);
}


} // namespace geometry
} // namespace nnrt
