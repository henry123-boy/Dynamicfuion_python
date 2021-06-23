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
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {



template<core::Device::DeviceType TDeviceType, typename TApplyBlendWarp>
void IntegrateWarped_Generic(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                             core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                             const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                             const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                             const float node_coverage, const int anchor_count, const int minimum_valid_anchor_count, const float depth_scale,
                             const float depth_max, TApplyBlendWarp&& blend_function) {
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
	core::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(block_indices.GetLength());
	int64_t node_count = warp_graph_nodes.GetLength();

	int64_t n_voxels = n_blocks * block_resolution3;
	// cosine value for each pixel
	cos_voxel_ray_to_normal = core::Tensor::Zeros(depth_tensor.GetShape(), core::Dtype::Float32, block_keys.GetDevice());

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
	core::kernel::CUDALauncher launcher;
#else
	core::kernel::CPULauncher launcher;
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
				if (!graph::FindAnchorsAndWeightsForPoint_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
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

				Eigen::Vector3f warped_voxel = blend_function(voxel_camera, anchor_indices, anchor_weights, node_indexer, anchor_count);
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

template<core::Device::DeviceType TDeviceType>
void IntegrateWarpedDQ(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                       core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                       const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                       const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& warp_graph_nodes,
                       const core::Tensor& node_dual_quaternion_transformations, float node_coverage, int anchor_count,
                       const int minimum_valid_anchor_count, float depth_scale, float depth_max) {

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
			}
	);
}

template<core::Device::DeviceType TDeviceType>
void IntegrateWarpedMat(const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
                        core::Tensor& cos_voxel_ray_to_normal, int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
                        const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
                        const core::Tensor& intrinsics, const core::Tensor& extrinsics, const core::Tensor& graph_nodes,
                        const core::Tensor& node_rotations, const core::Tensor& node_translations, const float node_coverage, const int anchor_count,
                        const int minimum_valid_anchor_count, const float depth_scale, const float depth_max) {
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
			}
	);
}

// struct Coord3i {
// 	Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
// 	bool operator==(const Coord3i& other) const {
// 		return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
// 	}
//
// 	int64_t x_;
// 	int64_t y_;
// 	int64_t z_;
// };
//
// struct Coord3iHash {
// 	size_t operator()(const Coord3i& k) const {
// 		static const size_t p0 = 73856093;
// 		static const size_t p1 = 19349669;
// 		static const size_t p2 = 83492791;
//
// 		return (static_cast<size_t>(k.x_) * p0) ^
// 		       (static_cast<size_t>(k.y_) * p1) ^
// 		       (static_cast<size_t>(k.z_) * p2);
// 	}
// };
//
// template<open3d::core::Device::DeviceType TDeviceType>
// void TouchWarpedMat(std::shared_ptr<open3d::core::Hashmap>& hashmap,
//                     const open3d::core::Tensor& points,
//                     open3d::core::Tensor& voxel_block_coords,
//                     int64_t voxel_grid_resolution,
//                     const open3d::core::Tensor& extrinsics,
//                     const open3d::core::Tensor& warp_graph_nodes,
//                     const open3d::core::Tensor& node_rotations,
//                     const open3d::core::Tensor& node_translations,
//                     float node_coverage,
//                     float voxel_size,
//                     float sdf_trunc) {
// 	int64_t resolution = voxel_grid_resolution;
// 	float block_size = voxel_size * resolution;
//
// 	int64_t n = points.GetLength();
// 	const float* pcd_ptr = static_cast<const float*>(points.GetDataPtr());
//
// #if defined(__CUDACC__)
// 	core::Device device = points.GetDevice();
// 	core::Tensor block_coordi({8 * n, 3}, core::Dtype::Int32, device);
// 	int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());
// 	core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32, device);
// 	int* count_ptr = static_cast<int*>(count.GetDataPtr());
// 	core::kernel::CUDALauncher launcher;
//
// 	#define floor_device(X) floorf(X)
// #else
// 	tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
// 	core::kernel::CPULauncher launcher;
//
// 	#define floor_device(X) std::floor(X)
// #endif
//
// 	launcher.LaunchGeneralKernel(
// 			n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
// 				float x = pcd_ptr[3 * workload_idx + 0];
// 				float y = pcd_ptr[3 * workload_idx + 1];
// 				float z = pcd_ptr[3 * workload_idx + 2];
//
// 				int xb_lo =
// 						static_cast<int>(floor_device((x - sdf_trunc) / block_size));
// 				int xb_hi =
// 						static_cast<int>(floor_device((x + sdf_trunc) / block_size));
// 				int yb_lo =
// 						static_cast<int>(floor_device((y - sdf_trunc) / block_size));
// 				int yb_hi =
// 						static_cast<int>(floor_device((y + sdf_trunc) / block_size));
// 				int zb_lo =
// 						static_cast<int>(floor_device((z - sdf_trunc) / block_size));
// 				int zb_hi =
// 						static_cast<int>(floor_device((z + sdf_trunc) / block_size));
//
// 				for (int xb = xb_lo; xb <= xb_hi; ++xb) {
// 					for (int yb = yb_lo; yb <= yb_hi; ++yb) {
// 						for (int zb = zb_lo; zb <= zb_hi; ++zb) {
// #if defined(__CUDACC__)
// 							int idx = atomicAdd(count_ptr, 1);
// 							block_coordi_ptr[3 * idx + 0] = xb;
// 							block_coordi_ptr[3 * idx + 1] = yb;
// 							block_coordi_ptr[3 * idx + 2] = zb;
// #else
// 							// set.emplace(xb, yb, zb);
// #endif
// 						}
// 					}
// 				}
// 			}
// 	);
//
// #if defined(__CUDACC__)
// 	int total_block_count = count.Item<int>();
// #else
// 	int total_block_count = set.size();
// #endif
//
// 	#undef floor_device
//
//
// 	if (total_block_count == 0) {
// 		utility::LogError(
// 				"[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
// 				"abort integration. Please check specified parameters, "
// 				"especially depth_scale and voxel_size");
// 	}
//
// #if defined(__CUDACC__)
// 	block_coordi = block_coordi.Slice(0, 0, total_block_count);
// 	core::Tensor block_addrs, block_masks;
// 	hashmap->Activate(block_coordi.Slice(0, 0, count.Item<int>()), block_addrs,
// 	                  block_masks);
// 	voxel_block_coords = block_coordi.IndexGet({block_masks});
// #else
// 	voxel_block_coords = core::Tensor({total_block_count, 3}, core::Dtype::Int32,points.GetDevice());
// 	int* block_coords_ptr = static_cast<int*>(voxel_block_coords.GetDataPtr());
// 	int count = 0;
// 	for (auto it = set.begin(); it != set.end(); ++it, ++count) {
// 		int64_t offset = count * 3;
// 		block_coords_ptr[offset + 0] = static_cast<int>(it->x_);
// 		block_coords_ptr[offset + 1] = static_cast<int>(it->y_);
// 		block_coords_ptr[offset + 2] = static_cast<int>(it->z_);
// 	}
// #endif
// }


} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt
