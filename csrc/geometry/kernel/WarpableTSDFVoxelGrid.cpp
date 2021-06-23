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

void IntegrateWarpedDQ(const core::Tensor& indices, const core::Tensor& block_keys, core::Tensor& block_values, core::Tensor& cos_voxel_ray_to_normal,
                       int64_t block_resolution, float voxel_size, float sdf_truncation_distance, const core::Tensor& depth_tensor,
                       const core::Tensor& color_tensor, const core::Tensor& depth_normals, const core::Tensor& intrinsics,
                       const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes, const core::Tensor& node_dual_quaternion_transformations,
                       float node_coverage, int anchor_count, const int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {
		IntegrateWarpedDQ<core::Device::DeviceType::CPU>(indices, block_keys, block_values,
		                                                 cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
		                                                 depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d, warp_graph_nodes,
		                                                 node_dual_quaternion_transformations, node_coverage, anchor_count, 0, depth_scale,
		                                                 depth_max);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedDQ<core::Device::DeviceType::CUDA>(indices, block_keys, block_values,
		                                                  cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
		                                                  depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d, warp_graph_nodes,
		                                                  node_dual_quaternion_transformations, node_coverage, anchor_count, 0, depth_scale,
		                                                  depth_max);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}

void IntegrateWarpedMat(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                        core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                        const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                        const core::Tensor& node_rotations, const core::Tensor& node_translations, const float node_coverage, const int anchor_count,
                        const int minimum_valid_anchor_count, const float depth_scale, const float depth_max) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	static const core::Device host("CPU:0");
	core::Tensor intrinsics_d =
			intrinsics.To(host, core::Dtype::Float64).Contiguous();
	core::Tensor extrinsics_d =
			extrinsics.To(host, core::Dtype::Float64).Contiguous();

	if (device_type == core::Device::DeviceType::CPU) {

		IntegrateWarpedMat<core::Device::DeviceType::CPU>(block_indices, block_keys, block_values,
		                                                  cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
		                                                  depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d, warp_graph_nodes,
		                                                  node_rotations, node_translations, node_coverage, anchor_count, 0, depth_scale, depth_max);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		IntegrateWarpedMat<core::Device::DeviceType::CUDA>(block_indices, block_keys, block_values,
		                                                   cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
		                                                   depth_tensor, color_tensor, depth_normals, intrinsics_d, extrinsics_d, warp_graph_nodes,
		                                                   node_rotations, node_translations, node_coverage, anchor_count, 0, depth_scale, depth_max);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}

}
//
// void TouchWarpedMat(std::shared_ptr<core::Hashmap>& hashmap, const core::Tensor& points, core::Tensor& voxel_block_coords,
// 					const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
//                     const core::Tensor& node_rotations, const core::Tensor& node_translations, float node_coverage,
//                     int64_t voxel_grid_resolution, float voxel_size, float sdf_trunc) {
//
// 	core::Device device = points.GetDevice();
//
// 	core::Device::DeviceType device_type = device.GetType();
// 	if (device_type == core::Device::DeviceType::CPU) {
// 		TouchWarpedMat<core::Device::DeviceType::CPU>(
// 				hashmap, points, voxel_block_coords,
// 				extrinsics, warp_graph_nodes, node_rotations, node_translations, node_coverage,
// 				voxel_grid_resolution, voxel_size, sdf_trunc
// 		);
// 	} else if (device_type == core::Device::DeviceType::CUDA) {
// #ifdef BUILD_CUDA_MODULE
// 		TouchWarpedMat<core::Device::DeviceType::CPU>(
// 				hashmap, points, voxel_block_coords,
// 				extrinsics, warp_graph_nodes, node_rotations, node_translations, node_coverage,
// 				voxel_grid_resolution, voxel_size, sdf_trunc
// 		);
// #else
// 		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
// #endif
// 	} else {
// 		utility::LogError("Unimplemented device");
// 	}
//
// }


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt