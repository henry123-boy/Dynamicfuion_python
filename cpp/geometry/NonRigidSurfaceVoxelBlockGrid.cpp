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
#include "geometry/NonRigidSurfaceVoxelBlockGrid.h"

#include <open3d/core/TensorKey.h>
#include <open3d/core/Device.h>
#include <open3d/t/geometry/Utility.h>

#include "geometry/kernel/NonRigidSurfaceVoxelBlockGrid.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry {


// TODO: adapt to adhere to the new allocation / integration function split, as in Open3D v >0.15.1
open3d::core::Tensor
NonRigidSurfaceVoxelBlockGrid::IntegrateNonRigid(const open3d::core::Tensor& block_coords, const WarpField& warp_field,
                                                 const open3d::t::geometry::Image& depth, const open3d::t::geometry::Image& color,
                                                 const open3d::core::Tensor& depth_normals,
                                                 const open3d::core::Tensor& depth_intrinsics, const open3d::core::Tensor& color_intrinsics,
                                                 const open3d::core::Tensor& extrinsics, float depth_scale,
                                                 float depth_max, float truncation_voxel_multiplier) {

	this->AssertInitialized();
	bool integrate_color = color.AsTensor().NumElements() > 0;

	o3tg::CheckBlockCoorinates(block_coords);
	o3tg::CheckDepthTensor(depth.AsTensor());
	if (integrate_color) {
		o3tg::CheckColorTensor(color.AsTensor());
	}
	o3tg::CheckIntrinsicTensor(depth_intrinsics);
	o3tg::CheckIntrinsicTensor(color_intrinsics);
	o3tg::CheckExtrinsicTensor(extrinsics);

	// Activate indicated blocks
	o3c::Tensor buffer_indices, masks;

	o3c::Tensor active_block_addresses;
	this->block_hashmap_->Activate(block_coords, buffer_indices, masks);
	this->block_hashmap_->GetActiveIndices(active_block_addresses);


	o3c::Tensor cos_voxel_ray_to_normal;

	o3tg::TensorMap block_value_map =
			VoxelBlockGrid::ConstructTensorMap(*block_hashmap_, name_attr_map_);

	kernel::voxel_grid::IntegrateNonRigid(active_block_addresses.To(o3c::Dtype::Int64), this->block_hashmap_->GetKeyTensor(),
	                                      block_value_map,
	                                      cos_voxel_ray_to_normal, this->block_resolution_, this->voxel_size_,
	                                      this->voxel_size_ * truncation_voxel_multiplier,
	                                      depth.AsTensor(), color.AsTensor(), depth_normals,
	                                      depth_intrinsics, color_intrinsics, extrinsics,
	                                      warp_field, depth_scale, depth_max);

	return cos_voxel_ray_to_normal;
}

o3c::Tensor NonRigidSurfaceVoxelBlockGrid::BufferCoordinatesOfInactiveNeighborBlocks(const o3c::Tensor& active_block_addresses) const {
	//TODO: shares most code with TSDFVoxelGrid::BufferRadiusNeighbors (DRY violation)
	auto block_hashmap = this->block_hashmap_;
	o3c::Tensor key_buffer_int3_tensor = block_hashmap->GetKeyTensor();

	o3c::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
			{active_block_addresses.To(o3c::Dtype::Int64)});
	int64_t current_block_count = active_keys.GetShape()[0];

	// Fill in radius nearest neighbors.
	o3c::Tensor neighbor_block_keys({27, current_block_count, 3}, o3c::Dtype::Int32, this->block_hashmap_->GetDevice());
	for (int neighbor_index = 0; neighbor_index < 27; neighbor_index++) {
		int dz = neighbor_index / 9;
		int dy = (neighbor_index % 9) / 3;
		int dx = neighbor_index % 3;
		o3c::Tensor dt = o3c::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1},
		                             {1, 3}, o3c::Dtype::Int32, this->GetDevice());
		neighbor_block_keys[neighbor_index] = active_keys + dt;
	}
	neighbor_block_keys = neighbor_block_keys.View({27 * current_block_count, 3});

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap->Find(neighbor_block_keys, neighbor_block_addresses, neighbor_mask);

	// ~ binary "or" to get the inactive address/coordinate mask instead of the active one
	neighbor_mask = neighbor_mask.LogicalNot();

	return neighbor_block_keys.GetItem(o3c::TensorKey::IndexTensor(neighbor_mask));
}

int64_t NonRigidSurfaceVoxelBlockGrid::ActivateSleeveBlocks() {
	o3c::Tensor active_indices;
	auto block_hashmap = this->block_hashmap_;
	block_hashmap->GetActiveIndices(active_indices);
	o3c::Tensor inactive_neighbor_of_active_blocks_coordinates =
			BufferCoordinatesOfInactiveNeighborBlocks(active_indices);

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap->Activate(inactive_neighbor_of_active_blocks_coordinates, neighbor_block_addresses, neighbor_mask);

	return inactive_neighbor_of_active_blocks_coordinates.GetShape()[0];
}


open3d::core::Tensor
NonRigidSurfaceVoxelBlockGrid::GetBoundingBoxesOfWarpedBlocks(const open3d::core::Tensor& block_keys, const WarpField& warp_field,
                                                              const open3d::core::Tensor& extrinsics) const {
	o3c::AssertTensorDtype(block_keys, o3c::Dtype::Int32);
	o3c::AssertTensorShape(block_keys, { utility::nullopt, 3 });

	o3c::Tensor bounding_boxes;
	kernel::voxel_grid::GetBoundingBoxesOfWarpedBlocks(bounding_boxes, block_keys, warp_field, this->GetVoxelSize(), this->block_resolution_,
	                                                   extrinsics);
	return bounding_boxes;
}

open3d::core::Tensor
NonRigidSurfaceVoxelBlockGrid::GetAxisAlignedBoxesIntersectingSurfaceMask(const open3d::core::Tensor& boxes, const open3d::t::geometry::Image& depth,
                                                                          const open3d::core::Tensor& intrinsics, float depth_scale,
                                                                          float depth_max, int downsampling_factor,
                                                                          float trunc_voxel_multiplier) const {
	open3d::t::geometry::CheckIntrinsicTensor(intrinsics);
	//TODO: figure out why we need NumElements > 0, make this check pass
	// open3d::t::geometry::CheckDepthTensor(depth.AsTensor());
	o3c::AssertTensorDtype(boxes, o3c::Dtype::Float32);
	o3c::AssertTensorShape(boxes, { utility::nullopt, 6 });

	o3c::Tensor mask;
	kernel::voxel_grid::GetAxisAlignedBoxesInterceptingSurfaceMask(mask, boxes, intrinsics, depth.AsTensor(), depth_scale, depth_max,
	                                                               downsampling_factor,
	                                                               this->voxel_size_ * trunc_voxel_multiplier);
	return mask;
}

open3d::core::Tensor
NonRigidSurfaceVoxelBlockGrid::FindBlocksIntersectingTruncationRegion(const open3d::t::geometry::Image& depth, const WarpField& warp_field,
                                                                      const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
                                                                      float depth_scale, float depth_max, float truncation_voxel_multiplier) const {
	AssertInitialized();
	o3tg::CheckDepthTensor(depth.AsTensor());
	o3tg::CheckIntrinsicTensor(intrinsics);
	o3tg::CheckExtrinsicTensor(extrinsics);

	static const o3c::Device host("CPU:0");
	o3c::Tensor intrinsics_host_double = intrinsics.To(host, o3c::Dtype::Float64).Contiguous();
	o3c::Tensor extrinsics_host_double = extrinsics.To(host, o3c::Dtype::Float64).Contiguous();

	// Query active blocks.
	o3c::Tensor active_block_addresses;
	this->block_hashmap_->GetActiveIndices(active_block_addresses);

	// Assert which neighboring inactive blocks need to be activated and activate them
	o3c::Tensor inactive_neighbor_block_coords = this->BufferCoordinatesOfInactiveNeighborBlocks(active_block_addresses);
	o3c::Tensor inactive_neighbor_bounding_boxes =
			this->GetBoundingBoxesOfWarpedBlocks(inactive_neighbor_block_coords, warp_field, extrinsics_host_double);

	o3c::Tensor inactive_neighbor_mask =
			this->GetAxisAlignedBoxesIntersectingSurfaceMask(inactive_neighbor_bounding_boxes, depth, intrinsics_host_double, depth_scale, depth_max,
			                                                 4);

	o3c::Tensor neighbor_coords_to_activate = inactive_neighbor_block_coords.GetItem(o3c::TensorKey::IndexTensor(inactive_neighbor_mask));
	return neighbor_coords_to_activate;
}

open3d::core::Tensor NonRigidSurfaceVoxelBlockGrid::ExtractVoxelValuesAndCoordinates() {
	o3c::Tensor active_block_indices;
	this->block_hashmap_->GetActiveIndices(active_block_indices);
	o3tg::TensorMap block_value_map =
			VoxelBlockGrid::ConstructTensorMap(*block_hashmap_, name_attr_map_);

	o3c::Tensor voxel_values;
	kernel::voxel_grid::ExtractVoxelValuesAndCoordinates(voxel_values,
	                                                     active_block_indices.To(o3c::Dtype::Int64), this->block_hashmap_->GetKeyTensor(),
	                                                     block_value_map, this->GetBlockResolution(), this->voxel_size_);

	return voxel_values;
}

open3d::core::Tensor NonRigidSurfaceVoxelBlockGrid::ExtractVoxelBlockCoordinates() {
	o3c::Tensor active_block_indices;
	this->block_hashmap_->GetActiveIndices(active_block_indices);
	return this->block_hashmap_->GetKeyTensor().GetItem(o3c::TensorKey::IndexTensor(active_block_indices.To(o3c::Int64))).To(o3c::Float32) *
	       this->block_resolution_ * this->voxel_size_;
}

open3d::core::Tensor NonRigidSurfaceVoxelBlockGrid::ExtractVoxelValuesAt(const open3d::core::Tensor& query_voxel_coordinates) {
	o3c::AssertTensorDtype(query_voxel_coordinates, o3c::Int32);
	o3c::AssertTensorShape(query_voxel_coordinates, { utility::nullopt, 3 });

	o3tg::TensorMap block_value_map =
			VoxelBlockGrid::ConstructTensorMap(*block_hashmap_, name_attr_map_);



	o3c::Tensor query_block_coordinates = query_voxel_coordinates / this->block_resolution_;
	o3c::Tensor query_block_indices, query_block_index_mask;
	this->block_hashmap_->Find(query_block_coordinates, query_block_indices, query_block_index_mask);

	o3c::Tensor active_query_coordinates = query_voxel_coordinates.GetItem(o3c::TensorKey::IndexTensor(query_block_index_mask));
	o3c::Tensor active_query_block_indices = query_block_indices.GetItem(o3c::TensorKey::IndexTensor(query_block_index_mask));


	o3c::Tensor voxel_values;
	kernel::voxel_grid::ExtractVoxelValuesAt(voxel_values, active_query_coordinates, active_query_block_indices,
											 this->block_hashmap_->GetKeyTensor(), block_value_map,
	                                         this->GetBlockResolution(), this->voxel_size_);

	return voxel_values;

}


} // namespace nnrt::geometry
