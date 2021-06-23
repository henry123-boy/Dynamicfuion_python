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


open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarpedDQ(const Image& depth, const Image& color,
                                         const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                         const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                         const core::Tensor& node_dual_quaternion_transformations,
                                         float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                         float depth_scale, float depth_max) {

	// note the difference from TSDFVoxelGrid::Integrate:
	// IntegrateWarpedDQ assumes that all of the relevant hash blocks have already been activated.

	if (depth.IsEmpty()) {
		utility::LogError(
				"[TSDFVoxelGrid] input depth is empty for integration.");
	}

	core::Tensor depth_tensor = depth.AsTensor().To(core::Dtype::Float32).Contiguous();
	core::Tensor color_tensor;

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

	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);

	core::Tensor block_values = block_hashmap_->GetValueTensor();


	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedDQ(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedDQ(const Image& depth, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                              const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                              const core::Tensor& node_dual_quaternion_transformations, float node_coverage,
                                                              int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedDQ(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_dual_quaternion_transformations,
	                         node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedMat(const Image& depth, const Image& color, const core::Tensor& depth_normals,
                                                               const core::Tensor& intrinsics, const core::Tensor& extrinsics,
                                                               const core::Tensor& warp_graph_nodes,
                                                               const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                                               float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                               float depth_scale, float depth_max) {
	// note the difference from TSDFVoxelGrid::Integrate:
	// IntegrateWarpedDQ assumes that all of the relevant hash blocks have already been activated.

	if (depth.IsEmpty()) {
		utility::LogError(
				"[TSDFVoxelGrid] input depth is empty for integration.");
	}

	// Downsample image to roughly estimate surfaces.

	float downsampling_factor = 0.5;
	auto depth_downsampled = depth.Resize(downsampling_factor, Image::InterpType::Linear);

	core::Tensor active_indices;
	block_hashmap_->GetActiveIndices(active_indices);
	core::Tensor inactive_neighbor_of_active_blocks_coordinates =
			BufferInactiveRadiusNeighborBlockCoordinates(active_indices);

	core::Tensor block_coords;
	// kernel::tsdf::TouchWarpedMat(point_hashmap_, pcd.GetPoints().Contiguous(),
	//                              block_coords, extrinsics, warp_graph_nodes,
	//                              node_rotations, node_translations, node_coverage,
	//                              block_resolution_, voxel_size_, sdf_trunc_);



	core::Tensor depth_tensor = depth.AsTensor().To(core::Dtype::Float32).Contiguous();
	core::Tensor color_tensor;

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


	// Query active blocks and their nearest neighbors to handle boundary cases.
	core::Tensor active_block_addresses;
	block_hashmap_->GetActiveIndices(active_block_addresses);

	core::Tensor block_values = block_hashmap_->GetValueTensor();


	core::Tensor cos_voxel_ray_to_normal;

	kernel::tsdf::IntegrateWarpedMat(
			active_block_addresses.To(core::Dtype::Int64), block_hashmap_->GetKeyTensor(), block_values,
			cos_voxel_ray_to_normal, block_resolution_, voxel_size_, sdf_trunc_,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes,
			node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
	);

	return cos_voxel_ray_to_normal;
}

open3d::core::Tensor WarpableTSDFVoxelGrid::IntegrateWarpedMat(const Image& depth, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                                               const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                                               const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                                               float node_coverage, int anchor_count, int minimum_valid_anchor_count,
                                                               float depth_scale, float depth_max) {
	Image empty_color;
	return IntegrateWarpedMat(depth, empty_color, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_rotations, node_translations,
	                          node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max);
}

open3d::core::Tensor WarpableTSDFVoxelGrid::BufferInactiveRadiusNeighborBlockCoordinates(const core::Tensor& active_block_addresses) {
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
	neighbor_mask = 1 - neighbor_mask;

	return keys_nb.GetItem(core::TensorKey::IndexTensor(neighbor_mask));
}


} // namespace geometry
} // namespace nnrt
