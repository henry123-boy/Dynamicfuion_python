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
#include "geometry/WarpableTSDFVoxelGrid.h"

#include <open3d/core/TensorKey.h>
#include <open3d/core/Device.h>
#include <geometry/kernel/Defines.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid_Analytics.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
using namespace open3d::t::geometry;

namespace nnrt::geometry {

o3c::Tensor WarpableTSDFVoxelGrid::ExtractVoxelCenters() {
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_centers;
	kernel::tsdf::ExtractVoxelCenters(
			active_block_indices.To(o3c::Dtype::Int64),
			block_hashmap->GetKeyTensor(), block_hashmap->GetValueTensor(),
			voxel_centers, this->GetBlockResolution(), this->GetVoxelSize());

	return voxel_centers;
}


open3d::core::Tensor WarpableTSDFVoxelGrid::ExtractTSDFValuesAndWeights() {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_values;
	kernel::tsdf::ExtractTSDFValuesAndWeights(
			active_block_indices.To(o3c::Dtype::Int64),
			block_hashmap->GetValueTensor(),
			voxel_values, this->GetBlockResolution());

	return voxel_values;
}

o3c::Tensor WarpableTSDFVoxelGrid::ExtractValuesInExtent(int min_voxel_x, int min_voxel_y, int min_voxel_z,
                                                         int max_voxel_x, int max_voxel_y, int max_voxel_z) {
	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_indices);

	o3c::Tensor voxel_values;
	kernel::tsdf::ExtractValuesInExtent(
			min_voxel_x, min_voxel_y, min_voxel_z,
			max_voxel_x, max_voxel_y, max_voxel_z,
			active_block_indices.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_hashmap->GetValueTensor(),
			voxel_values, this->GetBlockResolution());

	return voxel_values;
}

inline
static void PrepareDepthAndColorForIntegration(o3c::Tensor& depth_tensor, o3c::Tensor& color_tensor, const Image& depth, const Image& color,
                                               const std::unordered_map<std::string, o3c::Dtype>& attr_dtype_map_) {
	if (depth.IsEmpty()) {
		o3u::LogError(
				"[TSDFVoxelGrid] input depth is empty for integration.");
	}

	depth_tensor = depth.AsTensor().To(o3c::Dtype::Float32).Contiguous();

	if (color.IsEmpty()) {
		o3u::LogDebug(
				"[TSDFIntegrateWarped] color image is empty, perform depth "
				"integration only.");
	} else if (color.GetRows() == depth.GetRows() &&
	           color.GetCols() == depth.GetCols() && color.GetChannels() == 3) {
		if (attr_dtype_map_.count("color") != 0) {
			color_tensor = color.AsTensor().To(o3c::Dtype::Float32).Contiguous();
		} else {
			o3u::LogWarning(
					"[TSDFIntegrateWarped] color image is ignored since voxels do "
					"not contain colors.");
		}
	} else {
		o3u::LogWarning(
				"[TSDFIntegrateWarped] color image is ignored for the incompatible "
				"shape.");
	}
}




open3d::core::Tensor
WarpableTSDFVoxelGrid::IntegrateWarped(const Image& depth, const open3d::t::geometry::Image& color,
                                       const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& intrinsics,
                                       const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
									   float depth_scale, float depth_max) {
	// TODO note the difference from TSDFVoxelGrid::Integrate
	//  IntegrateWarpedEuclideanDQ currently assumes that all of the relevant hash blocks have already been activated. This will probably change in the future.

	o3c::AssertTensorDtype(intrinsics, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(extrinsics, o3c::Dtype::Float32);

	o3c::Tensor depth_tensor, color_tensor;
	PrepareDepthAndColorForIntegration(depth_tensor, color_tensor, depth, color, this->GetAttrDtypeMap());

	// Query active blocks and their nearest neighbors to handle boundary cases.
	o3c::Tensor active_block_addresses;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_block_addresses);
	o3c::Tensor block_values = block_hashmap->GetValueTensor();


	o3c::Tensor cos_voxel_ray_to_normal;

	static const o3c::Device host("CPU:0");
	o3c::Tensor intrinsics_host_double = intrinsics.To(host, o3c::Dtype::Float64).Contiguous();
	o3c::Tensor extrinsics_host_double = extrinsics.To(host, o3c::Dtype::Float64).Contiguous();

	kernel::tsdf::IntegrateWarped(active_block_addresses.To(o3c::Dtype::Int64), block_hashmap->GetKeyTensor(), block_values,
	                              cos_voxel_ray_to_normal, this->GetBlockResolution(), this->GetVoxelSize(), this->GetSDFTrunc(),
	                              depth_tensor, color_tensor, depth_normals, intrinsics_host_double, extrinsics_host_double, warp_field,
								  depth_scale, depth_max);

	return cos_voxel_ray_to_normal;
}

o3c::Tensor WarpableTSDFVoxelGrid::BufferCoordinatesOfInactiveNeighborBlocks(const o3c::Tensor& active_block_addresses) {
	//TODO: shares most code with TSDFVoxelGrid::BufferRadiusNeighbors (DRY violation)
	auto block_hashmap = this->GetBlockHashMap();
	o3c::Tensor key_buffer_int3_tensor = block_hashmap->GetKeyTensor();

	o3c::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
			{active_block_addresses.To(o3c::Dtype::Int64)});
	int64_t n = active_keys.GetShape()[0];

	// Fill in radius nearest neighbors.
	o3c::Tensor keys_nb({27, n, 3}, o3c::Dtype::Int32, this->GetDevice());
	for (int nb = 0; nb < 27; ++nb) {
		int dz = nb / 9;
		int dy = (nb % 9) / 3;
		int dx = nb % 3;
		o3c::Tensor dt = o3c::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1},
		                             {1, 3}, o3c::Dtype::Int32, this->GetDevice());
		keys_nb[nb] = active_keys + dt;
	}
	keys_nb = keys_nb.View({27 * n, 3});

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap->Find(keys_nb, neighbor_block_addresses, neighbor_mask);

	// ~ binary "or" to get the inactive address/coordinate mask instead of the active one
	neighbor_mask = neighbor_mask.LogicalNot();

	return keys_nb.GetItem(o3c::TensorKey::IndexTensor(neighbor_mask));
}

int64_t WarpableTSDFVoxelGrid::ActivateSleeveBlocks() {
	o3c::Tensor active_indices;
	auto block_hashmap = this->GetBlockHashMap();
	block_hashmap->GetActiveIndices(active_indices);
	o3c::Tensor inactive_neighbor_of_active_blocks_coordinates =
			BufferCoordinatesOfInactiveNeighborBlocks(active_indices);

	o3c::Tensor neighbor_block_addresses, neighbor_mask;
	block_hashmap->Activate(inactive_neighbor_of_active_blocks_coordinates, neighbor_block_addresses, neighbor_mask);

	return inactive_neighbor_of_active_blocks_coordinates.GetShape()[0];
}

std::ostream& operator<<(std::ostream& out, const WarpableTSDFVoxelGrid& grid) {
	// write header
	float voxel_size = grid.GetVoxelSize();
	float sdf_trunc = grid.GetSDFTrunc();
	int64_t block_resolution = grid.GetBlockResolution();
	int64_t block_count = grid.GetBlockCount();
	out.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));
	out.write(reinterpret_cast<const char*>(&sdf_trunc), sizeof(float));
	out.write(reinterpret_cast<const char*>(&block_resolution), sizeof(int64_t));
	out.write(reinterpret_cast<const char*>(&block_count), sizeof(int64_t));
	// write device type members (since no serialization is provided for that)
	int device_id = grid.GetDevice().GetID();
	out.write(reinterpret_cast<const char*>(&device_id), sizeof(int));
	o3c::Device::DeviceType device_type = grid.GetDevice().GetType();
	out.write(reinterpret_cast<const char*>(&device_type), sizeof(o3c::Device::DeviceType));
	//TODO
	throw std::runtime_error("Not implemented.");

	return out;
}


} // namespace nnrt::geometry
