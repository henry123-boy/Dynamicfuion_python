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

#include "WarpableTSDFVoxelGrid.h"
#include "utility/PlatformIndependence.h"
#include "geometry/DualQuaternion.h"
#include "geometry/kernel/Defines.h"
#include "geometry/kernel/GraphUtilitiesImpl.h"

#ifndef __CUDACC__

#include <tbb/concurrent_unordered_set.h>

#endif


using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {


template<o3c::Device::DeviceType TDeviceType, typename TApplyBlendWarp, typename TFindKNN>
void IntegrateWarped_Generic(const o3c::Tensor& block_indices, const o3c::Tensor& block_keys, o3c::Tensor& block_values,
                             o3c::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                             const o3c::Tensor& depth_tensor, const o3c::Tensor& color_tensor, const o3c::Tensor& depth_normals,
                             const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                             const float node_coverage, const int anchor_count, const int minimum_valid_anchor_count, const float depth_scale,
                             const float depth_max, TApplyBlendWarp&& blend_motion, TFindKNN&& find_knn) {
	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	float node_coverage_squared = node_coverage * node_coverage;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});
	TransformIndexer transform_indexer(intrinsics, extrinsics, 1.0);

	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("anchor_count is {}, but is required to satisfy 0 < anchor_count < {}", anchor_count, MAX_ANCHOR_COUNT);
	}

#if defined(__CUDACC__)
	o3c::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(block_indices.GetLength());
	int64_t node_count = warp_graph_nodes.GetLength();

	int64_t n_voxels = n_blocks * block_resolution3;
	// cosine value for each pixel
	cos_voxel_ray_to_normal = o3c::Tensor::Zeros(depth_tensor.GetShape(), o3c::Dtype::Float32, block_keys.GetDevice());

	// Data structure indexers
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
	// Motion graph indexer
	NDArrayIndexer node_indexer(warp_graph_nodes, 1);
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
#if defined(__CUDACC__)
	o3c::kernel::CUDALauncher launcher;
#else
	o3c::kernel::CPULauncher launcher;
#endif

	//  Go through voxels
//@formatter:off
	DISPATCH_BYTESIZE_TO_VOXEL(
			voxel_block_buffer_indexer.ElementByteSize(),
			[&]() {
				launcher.LaunchGeneralKernel(
						n_voxels,
						[=] OPEN3D_DEVICE (int64_t workload_idx) {
//@formatter:on
				// region ===================== COMPUTE VOXEL COORDINATE ================================
				// Natural index (0, N) ->
				//                    (workload_block_idx, voxel_index_in_block)
				int64_t block_index = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;


				// block_index -> x_block, y_block, z_block (in voxel hash blocks)
				int* block_key_ptr =
						block_keys_indexer.GetDataPtrFromCoord<int>(block_index);
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
				// endregion
				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];
				if (!find_knn(anchor_indices, anchor_weights, anchor_count,
				              minimum_valid_anchor_count, node_count,
				              voxel_global_metric, node_indexer, node_coverage_squared)) {
					return;
				}
				// endregion
				// region ===================== CONVERT VOXEL TO CAMERA SPACE, WARP IT, AND PROJECT TO IMAGE ============================

				// voxel world coordinate (in voxels) -> voxel camera coordinate (in meters)
				float x_voxel_camera, y_voxel_camera, z_voxel_camera;
				transform_indexer.RigidTransform(voxel_global_metric.x(), voxel_global_metric.y(), voxel_global_metric.z(),
				                                 &x_voxel_camera, &y_voxel_camera, &z_voxel_camera);
				Eigen::Vector3f voxel_camera(x_voxel_camera, y_voxel_camera, z_voxel_camera);

				Eigen::Vector3f warped_voxel = blend_motion(voxel_camera, anchor_indices, anchor_weights, node_indexer, anchor_count);
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
				auto voxel_pointer = voxel_block_buffer_indexer.GetDataPtrFromCoord<voxel_t>(
						x_voxel_local, y_voxel_local, z_voxel_local, block_index);

				auto u_rounded = static_cast<int64_t>(roundf(u_precise));
				auto v_rounded = static_cast<int64_t>(roundf(v_precise));

				float depth = (*depth_indexer.GetDataPtrFromCoord<float>(u_rounded, v_rounded)) / depth_scale;


				if (depth > 0.0f && depth < depth_max) {
					float psdf = depth - warped_voxel.z();

					Eigen::Vector3f view_direction = -warped_voxel;
					view_direction.normalize();

					// === compute normal ===
					auto normal_pointer = normals_indexer.GetDataPtrFromCoord<float>(u_rounded, v_rounded);
					Eigen::Vector3f normal(normal_pointer[0], normal_pointer[1], normal_pointer[2]);
					float cosine = normal.dot(view_direction);
					auto cosine_pointer = cosine_indexer.GetDataPtrFromCoord<float>(u_rounded, v_rounded);
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
						auto voxel_ptr = voxel_block_buffer_indexer.GetDataPtrFromCoord<voxel_t>(
								x_voxel_local, y_voxel_local, z_voxel_local, block_index);
						if (integrate_color) {
							auto color_ptr = color_indexer.GetDataPtrFromCoord<float>(u_rounded, v_rounded);
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

template<o3c::Device::DeviceType TDeviceType>
void IntegrateWarpedEuclideanDQ(const o3c::Tensor& block_indices, const o3c::Tensor& block_keys, o3c::Tensor& block_values,
                                o3c::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                const o3c::Tensor& depth_tensor, const o3c::Tensor& color_tensor, const o3c::Tensor& depth_normals,
                                const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics, const o3c::Tensor& warp_graph_nodes,
                                const o3c::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                                int minimum_valid_anchor_count, float depth_scale, float depth_max) {

	NDArrayIndexer node_transform_indexer(node_dual_quaternion_transformations, 1);

	IntegrateWarped_Generic<TDeviceType>(
			block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_coverage, anchor_count,
			minimum_valid_anchor_count, depth_scale, depth_max,
			[=] NNRT_DEVICE_WHEN_CUDACC
					(const Eigen::Vector3f& voxel_camera, const int* anchor_indices, const float* anchor_weights,
					 const NDArrayIndexer& node_indexer, const int anchor_count) {
				// *** linearly blend the anchor nodes' dual quaternions ***
				float coefficients[8];
				for (float& coefficient : coefficients) {
					coefficient = 0.0f;
				}
				for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
					int anchor_node_index = anchor_indices[i_anchor];
					if (anchor_node_index != -1) {
						float anchor_weight = anchor_weights[i_anchor];
						auto node_transform = node_transform_indexer.GetDataPtrFromCoord<float>(anchor_node_index);
						for (int i_coefficient = 0; i_coefficient < 8; i_coefficient++) {
							coefficients[i_coefficient] += anchor_weight * node_transform[i_coefficient];
						}
					}
				}
				Eigen::DualQuaternion<float> voxel_transformation(
						Eigen::Quaternion<float>(coefficients[0], coefficients[1], coefficients[2], coefficients[3]),
						Eigen::Quaternion<float>(coefficients[4], coefficients[5], coefficients[6], coefficients[7])
				);

				voxel_transformation.normalize();
				return voxel_transformation.transformPoint(voxel_camera);
			},
			[=] NNRT_DEVICE_WHEN_CUDACC(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
			                            const int minimum_valid_anchor_count,
			                            const int node_count, const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
			                            const float node_coverage_squared){
				return graph::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
				                                                                            minimum_valid_anchor_count, node_count,
				                                                                            point, node_indexer, node_coverage_squared);
			}
	);
}

template<o3c::Device::DeviceType TDeviceType>
void IntegrateWarpedEuclideanMat(const o3c::Tensor& block_indices, const o3c::Tensor& block_keys, o3c::Tensor& block_values,
                                 o3c::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                                 const o3c::Tensor& depth_tensor, const o3c::Tensor& color_tensor, const o3c::Tensor& depth_normals,
                                 const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics, const o3c::Tensor& graph_nodes,
                                 const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations, float node_coverage, int anchor_count,
                                 int minimum_valid_anchor_count, float depth_scale, float depth_max) {
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);

	IntegrateWarped_Generic<TDeviceType>(
			block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, graph_nodes, node_coverage, anchor_count, minimum_valid_anchor_count,
			depth_scale, depth_max,
			[=] NNRT_DEVICE_WHEN_CUDACC
					(const Eigen::Vector3f& voxel_camera, const int* anchor_indices, const float* anchor_weights,
					 const NDArrayIndexer& node_indexer, const int anchor_count) {
				Eigen::Vector3f warped_voxel(0.f, 0.f, 0.f);

				for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
					int anchor_node_index = anchor_indices[i_anchor];
					if (anchor_node_index != -1) {
						float anchor_weight = anchor_weights[i_anchor];
						auto node_rotation_data = node_rotation_indexer.GetDataPtrFromCoord<float>(anchor_node_index);
						auto node_translation_data = node_translation_indexer.GetDataPtrFromCoord<float>(anchor_node_index);
						Eigen::Matrix3f node_rotation(node_rotation_data);
						Eigen::Vector3f node_translation(node_translation_data);
						auto node_pointer = node_indexer.GetDataPtrFromCoord<float>(anchor_node_index);
						Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
						warped_voxel += anchor_weight * (node + node_rotation * (voxel_camera - node) + node_translation);
					}
				}
				return warped_voxel;
			},
			[=] NNRT_DEVICE_WHEN_CUDACC(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
			                            const int minimum_valid_anchor_count,
			                            const int node_count, const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
			                            const float node_coverage_squared){
				return graph::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
				                                                                            minimum_valid_anchor_count, node_count,
				                                                                            point, node_indexer, node_coverage_squared);
			}
	);
}

// inline
// NNRT_DEVICE_WHEN_CUDACC
// void ComputeVoxelHashBlockCorners()


template<o3c::Device::DeviceType TDeviceType>
void DetermineWhichBlocksToActivateWithWarp(o3c::Tensor& blocks_to_activate_mask, const o3c::Tensor& candidate_block_coordinates,
                                            const o3c::Tensor& depth_downsampled, const o3c::Tensor& intrinsics_downsampled,
                                            const o3c::Tensor& extrinsics, const o3c::Tensor& graph_nodes,
                                            const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                                            float node_coverage,
                                            int64_t block_resolution, float voxel_size, float sdf_truncation_distance) {
	auto candidate_block_count = candidate_block_coordinates.GetLength();
	blocks_to_activate_mask = o3c::Tensor({candidate_block_count}, o3c::Dtype::Bool, candidate_block_coordinates.GetDevice());

	NDArrayIndexer node_indexer(graph_nodes, 1);
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);
	NDArrayIndexer downsampled_depth_indexer(depth_downsampled, 2);

	// intermediate result storage
	o3c::Tensor candidate_block_corners({candidate_block_count * 8, 3}, o3c::Dtype::Float32, candidate_block_coordinates.GetDevice());
	NDArrayIndexer 	candidate_block_corner_indexer(candidate_block_corners, 1);
	TransformIndexer transform_indexer(intrinsics_downsampled, extrinsics, 1.0);

	//TODO
// #if defined(__CUDACC__)
// 	o3c::CUDACachedMemoryManager::ReleaseCache();
// #endif
// #if defined(__CUDACC__)
// 	o3c::kernel::CUDALauncher launcher;
// #else
// 	o3c::kernel::CPULauncher launcher;
// #endif
	//TODO
	// launcher.LaunchGeneralKernel(
	// 		candidate_block_count,
	// 		[=] OPEN3D_DEVICE {
	//
	// 		}
	// );


}


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt
