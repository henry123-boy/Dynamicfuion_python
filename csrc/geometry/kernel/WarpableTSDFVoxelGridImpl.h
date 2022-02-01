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
#include <cmath>

#include <Eigen/Geometry>

#include <open3d/core/Tensor.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/t/geometry/kernel/TSDFVoxel.h>
#include <open3d/t/geometry/kernel/TSDFVoxelGrid.h>

#include "core/PlatformIndependence.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/Defines.h"
#include "geometry/kernel/WarpUtilities.h"

#ifndef __CUDACC__

#include <tbb/concurrent_unordered_set.h>

#endif


using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;

namespace nnrt::geometry::kernel::tsdf {

template<open3d::core::Device::DeviceType TDeviceType>
void IntegrateWarped(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys, open3d::core::Tensor& block_values,
                     open3d::core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                     const open3d::core::Tensor& depth_tensor, const open3d::core::Tensor& color_tensor, const open3d::core::Tensor& depth_normals,
                     const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field,
                     float depth_scale, float depth_max) {
	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	float node_coverage_squared = warp_field.node_coverage * warp_field.node_coverage;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});
	TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0);

	int anchor_count = warp_field.anchor_count;
	int minimum_valid_anchor_count = warp_field.minimum_valid_anchor_count;
	int n_blocks = static_cast<int>(block_indices.GetLength());
	int64_t node_count = warp_field.nodes.GetLength();

	int64_t n_voxels = n_blocks * block_resolution3;
	// cosine value for each pixel
	cos_voxel_ray_to_normal = o3c::Tensor::Zeros(depth_tensor.GetShape(), o3c::Dtype::Float32, block_keys.GetDevice());

	// Data structure indexers
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
	// Motion graph indexer
	NDArrayIndexer node_indexer(warp_field.nodes, 1);
	NDArrayIndexer node_rotation_indexer(warp_field.rotations, 1);
	NDArrayIndexer node_translation_indexer(warp_field.translations, 1);
	// Image indexers
	NDArrayIndexer depth_indexer(depth_tensor, 2);
	NDArrayIndexer cosine_indexer(cos_voxel_ray_to_normal, 2);
	NDArrayIndexer normals_indexer(depth_normals, 2);
	// Optional color integration
	NDArrayIndexer color_indexer;
	bool integrate_color = false;
	if (color_tensor.NumElements() != 0) {
		color_indexer = NDArrayIndexer(color_tensor, 2);
		integrate_color = true;
	}

	// Plain array that does not require indexers
	const auto* indices_ptr = block_indices.GetDataPtr<int64_t>();

	//  Go through voxels
//@formatter:off
	DISPATCH_BYTESIZE_TO_VOXEL(
			voxel_block_buffer_indexer.ElementByteSize(),
			[&]() {
				open3d::core::ParallelFor(
						depth_tensor.GetDevice(),n_voxels,
						[=] OPEN3D_DEVICE (int64_t workload_idx) {
//@formatter:on
				// region ===================== COMPUTE VOXEL COORDINATE & CAMERA COORDINATE ================================
				// Natural index (0, N) ->
				//                    (workload_block_idx, voxel_index_in_block)
				int64_t block_index = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;


				// block_index -> x_block, y_block, z_block (in voxel hash blocks)
				int* block_key_ptr =
						block_keys_indexer.GetDataPtr<int>(block_index);
				auto x_block = static_cast<int64_t>(block_key_ptr[0]);
				auto y_block = static_cast<int64_t>(block_key_ptr[1]);
				auto z_block = static_cast<int64_t>(block_key_ptr[2]);

				// voxel_idx -> x_voxel_local, y_voxel_local, z_voxel_local (in voxels)
				int64_t x_voxel_local, y_voxel_local, z_voxel_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block, &x_voxel_local, &y_voxel_local, &z_voxel_local);

				// at this point, (x_voxel, y_voxel, z_voxel) hold local
				// in-block coordinates. Compute the global voxel coordinates (in voxels, then meters)
				Eigen::Vector3f voxel_global(x_block * block_resolution + x_voxel_local,
				                             y_block * block_resolution + y_voxel_local,
				                             z_block * block_resolution + z_voxel_local);
				Eigen::Vector3f voxel_global_metric = voxel_global * voxel_size;

				// voxel world coordinate (in voxels) -> voxel camera coordinate (in meters)
				float x_voxel_camera, y_voxel_camera, z_voxel_camera;
				transform_indexer.RigidTransform(voxel_global_metric.x(), voxel_global_metric.y(), voxel_global_metric.z(),
				                                 &x_voxel_camera, &y_voxel_camera, &z_voxel_camera);
				// endregion
				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];
				if (!warp::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, node_count, voxel_global_metric, node_indexer,
						node_coverage_squared
				)) {
					return;
				}
				// endregion
				// region ===================== WARP CAMERA-SPACE VOXEL AND PROJECT TO IMAGE ============================
				Eigen::Vector3f voxel_camera(x_voxel_camera, y_voxel_camera, z_voxel_camera);
				Eigen::Vector3f warped_voxel(0.f, 0.f, 0.f);

				warp::BlendWarp(warped_voxel, anchor_indices, anchor_weights, anchor_count, node_indexer,
				                node_rotation_indexer, node_translation_indexer, voxel_camera);

				if (warped_voxel.z() < 0) {
					// voxel is behind camera
					return;
				}
				// coordinate in image (in pixels)
				float u_precise, v_precise;
				transform_indexer.Project(warped_voxel.x(), warped_voxel.y(), warped_voxel.z(), &u_precise, &v_precise);
				if (!depth_indexer.InBoundary(u_precise, v_precise)) {
					return;
				}
				// endregion
				// region ===================== SAMPLE IMAGES AND COMPUTE THE ACTUAL TSDF & COLOR UPDATE =======================
				auto voxel_pointer = voxel_block_buffer_indexer.GetDataPtr<voxel_t>(
						x_voxel_local, y_voxel_local, z_voxel_local, block_index);

				auto u_rounded = static_cast<int64_t>(roundf(u_precise));
				auto v_rounded = static_cast<int64_t>(roundf(v_precise));

				float depth = (*depth_indexer.GetDataPtr<float>(u_rounded, v_rounded)) / depth_scale;

				if (depth > 0.0f && depth < depth_max) {
					float psdf = depth - warped_voxel.z();

					Eigen::Vector3f view_direction = -warped_voxel;
					view_direction.normalize();

					// === compute normal ===
					auto normal_pointer = normals_indexer.GetDataPtr<float>(u_rounded, v_rounded);
					Eigen::Vector3f normal(normal_pointer[0], normal_pointer[1], normal_pointer[2]);
					float cosine = view_direction.dot(normal);
					auto cosine_pointer = cosine_indexer.GetDataPtr<float>(u_rounded, v_rounded);
					*cosine_pointer = cosine;

					/*
					 * Points behind the surface are disregarded: these will have a projective signed distance
					 * below the negative truncation distance.
					 * Also, if the angle between view direction and surface normal is too oblique,
					 * we assume depth reading is unreliable
					 * */
					if (psdf > -sdf_truncation_distance && cosine > 0.5f) {
						float tsdf_normalized =
								(psdf < sdf_truncation_distance ? psdf : sdf_truncation_distance) / sdf_truncation_distance;
						auto voxel_ptr = voxel_block_buffer_indexer.GetDataPtr<voxel_t>(
								x_voxel_local, y_voxel_local, z_voxel_local, block_index);

						if (integrate_color) {
							auto color_ptr = color_indexer.GetDataPtr<float>(u_rounded, v_rounded);
							voxel_ptr->Integrate(tsdf_normalized, color_ptr[0], color_ptr[1], color_ptr[2]);
						} else {
							voxel_ptr->Integrate(tsdf_normalized);
						}
					}
				}
				// endregion
			} // end element_kernel
				); // end LaunchGeneralKernel call
			} // end lambda
	); // end DISPATCH_BYTESIZE_TO_VOXEL macro call
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}


// inline
// NNRT_DEVICE_WHEN_CUDACC
// void ComputeVoxelHashBlockCorners()


template<o3c::Device::DeviceType TDeviceType>
void DetermineWhichBlocksToActivateWithWarp(
		o3c::Tensor& blocks_to_activate_mask, const o3c::Tensor& candidate_block_coordinates,
		const o3c::Tensor& depth_downsampled, const o3c::Tensor& intrinsics_downsampled,
		const o3c::Tensor& extrinsics, const o3c::Tensor& graph_nodes, const o3c::Tensor& graph_edges,
		const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
		float node_coverage, int64_t block_resolution, float voxel_size, float sdf_truncation_distance
) {
	auto candidate_block_count = candidate_block_coordinates.GetLength();
	blocks_to_activate_mask = o3c::Tensor({candidate_block_count}, o3c::Dtype::Bool, candidate_block_coordinates.GetDevice());

	NDArrayIndexer node_indexer(graph_nodes, 1);
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);
	NDArrayIndexer downsampled_depth_indexer(depth_downsampled, 2);

	// intermediate result storage
	o3c::Tensor candidate_block_corners({candidate_block_count * 8, 3}, o3c::Dtype::Float32, candidate_block_coordinates.GetDevice());
	NDArrayIndexer candidate_block_corner_indexer(candidate_block_corners, 1);
	TransformIndexer transform_indexer(intrinsics_downsampled, extrinsics, 1.0);

	open3d::utility::LogError("Sorry, function not implemented.");
	//TODO
// #if defined(__CUDACC__)
// 	o3c::CUDACachedMemoryManager::ReleaseCache();
// #endif
// #if defined(__CUDACC__)
// 	namespace launcher = o3c::kernel::cuda_launcher;
// #else
// 	namespace launcher = o3c::kernel::cpu_launcher;
// #endif
	//TODO
	// launcher::ParallelFor(
	// 		candidate_block_count,
	// 		[=] OPEN3D_DEVICE {
	//
	// 		}
	// );


}


} // namespace nnrt::geometry::kernel::tsdf
