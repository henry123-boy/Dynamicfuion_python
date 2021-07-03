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
#include <open3d/core/Tensor.h>
#include <open3d/utility/Console.h>
#include <open3d/core/hashmap/Hashmap.h>

#include "geometry/kernel/WarpableTSDFVoxelGrid.h"

using namespace open3d;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

void IntegrateWarpedEuclideanDQ(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values, core::Tensor& cos_voxel_ray_to_normal,
                                int64_t block_resolution, float voxel_size, float sdf_truncation_distance, const core::Tensor& depth_tensor,
                                const core::Tensor& color_tensor, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                                const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes, const core::Tensor& node_dual_quaternion_transformations,
                                float node_coverage, int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	//TODO: establish why we have the conversion here / confirm it is needed
	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {
		IntegrateWarpedEuclideanDQ<core::Device::DeviceType::CPU>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedEuclideanDQ<core::Device::DeviceType::CUDA>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}

void IntegrateWarpedEuclideanMat(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                                 core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                 const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                                 const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                 const core::Tensor& node_rotations, const core::Tensor& node_translations, float node_coverage, int anchor_count,
                                 int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {

		IntegrateWarpedEuclideanMat<core::Device::DeviceType::CPU>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
		);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedEuclideanMat<core::Device::DeviceType::CUDA>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count, depth_scale, depth_max
		);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}


void IntegrateWarpedShortestPathDQ(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                                   core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                   const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                                   const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                   const core::Tensor& warp_graph_edges, const core::Tensor& node_dual_quaternion_transformations,
                                   float node_coverage, int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	//TODO: establish why we have the conversion here / confirm it is needed
	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {
		IntegrateWarpedShortestPathDQ<core::Device::DeviceType::CPU>(
				block_indices, block_keys, block_values,cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, warp_graph_edges, node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedShortestPathDQ<core::Device::DeviceType::CUDA>(
				block_indices, block_keys, block_values,cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, warp_graph_edges, node_dual_quaternion_transformations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}

void IntegrateWarpedShortestPathMat(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                                    core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                    const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                                    const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                                    const core::Tensor& warp_graph_edges, const core::Tensor& node_rotations, const core::Tensor& node_translations,
                                    float node_coverage, int anchor_count, int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {
		IntegrateWarpedShortestPathMat<core::Device::DeviceType::CPU>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, warp_graph_edges, node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedShortestPathMat<core::Device::DeviceType::CUDA>(
				block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
				depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d,
				warp_graph_nodes, warp_graph_edges, node_rotations, node_translations, node_coverage, anchor_count, minimum_valid_anchor_count,
				depth_scale, depth_max
		);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}


void DetermineWhichBlocksToActivateWithWarp(core::Tensor& blocks_to_activate_mask, const core::Tensor& candidate_block_coordinates,
                                            const core::Tensor& depth_downsampled, const core::Tensor& intrinsics_downsampled,
                                            const core::Tensor& extrinsics, const core::Tensor& graph_nodes, const core::Tensor& node_rotations,
                                            const core::Tensor& node_translations, float node_coverage, int64_t block_resolution, float voxel_size,
                                            float sdf_truncation_distance) {
	core::Device device = candidate_block_coordinates.GetDevice();
	core::Device::DeviceType device_type = device.GetType();
	if (device_type == core::Device::DeviceType::CPU) {

		DetermineWhichBlocksToActivateWithWarp<core::Device::DeviceType::CPU>(
				blocks_to_activate_mask, candidate_block_coordinates, depth_downsampled,
				intrinsics_downsampled, extrinsics, graph_nodes, node_rotations,
				node_translations, node_coverage, block_resolution, voxel_size, sdf_truncation_distance
		);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		DetermineWhichBlocksToActivateWithWarp<core::Device::DeviceType::CUDA>(
				blocks_to_activate_mask, candidate_block_coordinates, depth_downsampled,
				intrinsics_downsampled, extrinsics, graph_nodes, node_rotations,
				node_translations, node_coverage, block_resolution, voxel_size, sdf_truncation_distance
		);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt