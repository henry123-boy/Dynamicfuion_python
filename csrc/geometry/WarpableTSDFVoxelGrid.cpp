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
#include <open3d/core/TensorKey.h>
#include <utility>
#include <geometry/kernel/Defines.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid_Analytics.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt {
namespace geometry {

core::Tensor WarpableTSDFVoxelGrid::ExtractVoxelCenters() {
	core::Tensor active_block_indices;
	block_hashmap_->GetActiveIndices(active_block_indices);

	core::Tensor voxel_centers;
	kernel::tsdf::ExtractVoxelCenters(
			active_block_indices.To(core::Dtype::Int64),
			block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
			voxel_centers, block_resolution_, voxel_size_);

	return voxel_centers;
}


open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractTSDFValuesAndWeights() {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_indices;
	block_hashmap_->GetActiveIndices(active_block_indices);

	core::Tensor voxel_values;
	kernel::tsdf::ExtractTSDFValuesAndWeights(
			active_block_indices.To(core::Dtype::Int64),
			block_hashmap_->GetValueTensor(),
			voxel_values, block_resolution_);

	return voxel_values;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractValuesInExtent(int min_voxel_x, int min_voxel_y, int min_voxel_z,
                                                                  int max_voxel_x, int max_voxel_y, int max_voxel_z) {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_indices;
	block_hashmap_->GetActiveIndices(active_block_indices);

	core::Tensor voxel_values;
	kernel::tsdf::ExtractValuesInExtent(
			min_voxel_x, min_voxel_y, min_voxel_z,
			max_voxel_x, max_voxel_y, max_voxel_z,
			active_block_indices.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
			voxel_values, block_resolution_);

	return voxel_values;
}

inline
static void PrepareDepthAndColorForIntegration(core::Tensor& depth_tensor, core::Tensor& color_tensor, const Image& depth, const Image& color,
											   const std::unordered_map<std::string, core::Dtype>& attr_dtype_map_){
	if (depth.IsEmpty()) {
		utility::LogError(
				"[TSDFVoxelGrid] input depth is empty for integration.");
	}

	depth_tensor = depth.AsTensor().To(core::Dtype::Float32).Contiguous();

	if (color.IsEmpty()) {
		utility::LogDebug(
				"[TSDFIntegrateWarped] color image is empty, perform depth "
				"integration only.");
	} else if (color.GetRows() == depth.GetRows() &&
	           color.GetCols() == depth.GetCols() && color.GetChannels() == 3) {
		if (attr_dtype_map_.count("color") != 0) {
			color_tensor = color.AsTensor().To(core::Dtype::Float32).Contiguous();
		} else {
			utility::LogWarning(
					"[TSDFIntegrateWarped] color image is ignored since voxels do "
					"not contain colors.");
		}
	} else {
		utility::LogWarning(
				"[TSDFIntegrateWarped] color image is ignored for the incompatible "
				"shape.");
	}
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanDQ(const Image& depth, const Image& color,
                                                  const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                  const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                  const core::Tensor& node_dual_quaternion_transformations,
                                                  float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                  float depth_scale, float depth_max) {

	intrinsics.AssertDtype(core::Dtype::Float32);
	extrinsics.AssertDtype(core::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("`anchor_count` is {}, but is required to satisfy 0 < anchor_count <= {}", anchor_count, MAX_ANCHOR_COUNT);
	}
	if (minimum_valid_anchor_count < 0 || minimum_valid_anchor_count > anchor_count){
		utility::LogError("`minimum_valid_anchor_count` is {}, but is required to satisfy 0 < minimum_valid_anchor_count <= {} ",
						  minimum_valid_anchor_count, anchor_count);
	}

	// TODO note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ currently assumes that all of the relevant hash blocks have already been activated. This will probably change in the future.
	core::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->attr_dtype_map_);

	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);
	core::Tensor block_values = block_hashmap_->GetValueTensor();

	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedEuclideanDQ(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanDQ(const Image& depth, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                                       const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                                       const core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                                       int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedEuclideanDQ(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
	                                  node_dual_quaternion_transformations,
	                                  node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanMat(const Image& depth, const Image& color, const core::Tensor& depth_normals,
                                                                        const core::Tensor& intrinsics, const core::Tensor& extrinsics,
                                                                        const core::Tensor& warp_graph_nodes,
                                                                        const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                                                        float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                                        float depth_scale, float depth_max) {
	// TODO note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ currently assumes that all of the relevant hash blocks have already been activated. This will probably change in the future.

	intrinsics.AssertDtype(core::Dtype::Float32);
	extrinsics.AssertDtype(core::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}


	// float downsampling_factor = 0.5;
	// auto depth_downsampled = depth.Resize(downsampling_factor, Image::InterpType::Linear);
	// core::Tensor intrinsics_downsampled = intrinsics * downsampling_factor;
	//
	// core::Tensor active_indices;
	// block_hashmap_->GetActiveIndices(active_indices);
	// core::Tensor coordinates_of_inactive_neighbors_of_active_blocks =
	// 		BufferCoordinatesOfInactiveNeighborBlocks(active_indices);
	// core::Tensor blocks_to_activate_mask;
	// kernel::tsdf::DetermineWhichBlocksToActivateWithWarp(
	// 		blocks_to_activate_mask,
	// 		coordinates_of_inactive_neighbors_of_active_blocks,
	// 		depth_downsampled.AsTensor().To(core::Dtype::Float32).Contiguous(),
	// 		intrinsics_downsampled, extrinsics, warp_graph_nodes, node_rotations, node_translations,
	// 		node_coverage, block_resolution_, voxel_size_, sdf_trunc_);
	// core::Tensor block_coords;

	core::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->attr_dtype_map_);

	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);
	core::Tensor block_values = block_hashmap_->GetValueTensor();


	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedEuclideanMat(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}



open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedEuclideanMat(const Image& depth, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                                        const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                                        const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                                                        float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                                        float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedEuclideanMat(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_rotations, node_translations,
	                                   node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathDQ(const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color,
                                                     const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& intrinsics,
                                                     const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                                                     const open3d::core::Tensor& warp_graph_edges,
                                                     const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                     int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {

	// TODO: note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ assumes that all of the relevant hash blocks have already been activated.
	intrinsics.AssertDtype(core::Dtype::Float32);
	extrinsics.AssertDtype(core::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

	core::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->attr_dtype_map_);

	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);

	core::Tensor block_values = block_hashmap_->GetValueTensor();


	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedShortestPathDQ(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
			node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathDQ(const open3d::t::geometry::Image& depth, const open3d::core::Tensor& depth_normals,
                                                     const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                                                     const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                                                     const open3d::core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                     int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedShortestPathDQ(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
	                                     node_dual_quaternion_transformations,
	                                     node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathMat(const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color,
                                                      const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& intrinsics,
                                                      const open3d::core::Tensor& extrinsics, const open3d::core::Tensor& warp_graph_nodes,
                                                      const open3d::core::Tensor& warp_graph_edges, const open3d::core::Tensor& node_rotations,
                                                      const open3d::core::Tensor& node_translations, float node_coverage, int anchor_count,
                                                      int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	// TODO: note the difference from TSDFVoxelGrid::Integrate:
	//  IntegrateWarpedEuclideanDQ assumes that all of the relevant hash blocks have already been activated.

	intrinsics.AssertDtype(core::Dtype::Float32);
	extrinsics.AssertDtype(core::Dtype::Float32);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

	core::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->attr_dtype_map_);


	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);

	core::Tensor block_values = block_hashmap_->GetValueTensor();


	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedShortestPathMat(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, warp_graph_edges,
			node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}


open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedShortestPathMat(const open3d::t::geometry::Image& depth, const open3d::core::Tensor& depth_normals,
                                                      const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                                                      const open3d::core::Tensor& warp_graph_nodes, const open3d::core::Tensor& warp_graph_edges,
                                                      const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                                                      float node_coverage, int anchor_count, int minimum_valid_anchor_count, float depth_scale,
                                                      float depth_max) {
	Image empty_color;
	return IntegrateWarpedShortestPathMat(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
	                                      warp_graph_edges, node_rotations, node_translations,
	                                      node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor WarpableTSDFVoxelGrid::BufferCoordinatesOfInactiveNeighborBlocks(const core::Tensor& active_block_addresses) {
	//TODO: shares most code with TSDFVoxelGrid::BufferRadiusNeighbors (DRY violation)
	core::Tensor key_buffer_int3_tensor = block_hashmap_->GetKeyTensor();

	core::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
			{active_block_addresses.To(core::Dtype::Int64)});
	int64_t n = active_keys.GetShape()[0];

	// Fill in radius nearest neighbors.
	core::Tensor keys_nb({27, n, 3}, core::Dtype::Int32, device_);
	for (int nb = 0; nb < 27; ++nb) {
		int dz = nb / 9;
		int dy = (nb % 9) / 3;
		int dx = nb % 3;
		core::Tensor dt = core::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1},
		                               {1, 3}, core::Dtype::Int32, device_);
		keys_nb[nb] = active_keys + dt;
	}
	keys_nb = keys_nb.View({27 * n, 3});

	core::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap_->Find(keys_nb, neighbor_block_addresses, neighbor_mask);

	// ~ binary "or" to get the inactive address/coordinate mask instead of the active one
	neighbor_mask = neighbor_mask.LogicalNot();

	return keys_nb.GetItem(core::TensorKey::IndexTensor(neighbor_mask));
}

int64_t WarpableTSDFVoxelGrid::ActivateSleeveBlocks() {
	core::Tensor active_indices;
	block_hashmap_->GetActiveIndices(active_indices);
	core::Tensor inactive_neighbor_of_active_blocks_coordinates =
			BufferCoordinatesOfInactiveNeighborBlocks(active_indices);

	core::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap_->Activate(inactive_neighbor_of_active_blocks_coordinates, neighbor_block_addresses, neighbor_mask);

	return inactive_neighbor_of_active_blocks_coordinates.GetShape()[0];
}

} // namespace geometry
} // namespace nnrt
