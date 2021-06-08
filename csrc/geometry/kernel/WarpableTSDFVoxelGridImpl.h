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

#include "utility/PlatformIndependence.h"
#include "WarpableTSDFVoxelGrid.h"
#include "geometry/DualQuaternion.h"


#define MAX_ANCHOR_COUNT 8
#define MINIMUM_VALID_ANCHOR_COUNT 3

using namespace open3d;
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

#if defined(__CUDACC__)
void ExtractVoxelCentersCUDA
#else

void ExtractVoxelCentersCPU
#endif
		(const core::Tensor& indices,
		 const core::Tensor& block_keys,
		 const core::Tensor& block_values,
		 core::Tensor& voxel_centers,
		 int64_t block_resolution, float voxel_size) {

	int64_t resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});

	// Output
#if defined(__CUDACC__)
	core::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(indices.GetLength());

	int64_t n_voxels = n_blocks * resolution3;
	// each voxel center will need three coordinates: n_voxels x 3
	voxel_centers = core::Tensor({n_voxels, 3}, core::Dtype::Float32,
	                             block_keys.GetDevice());

	// Real data indexers
	NDArrayIndexer voxel_centers_indexer(voxel_centers, 1);
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	// Plain array that does not require indexers
	const int64_t* indices_ptr = indices.GetDataPtr<int64_t>();

#if defined(__CUDACC__)
	core::kernel::CUDALauncher launcher;
#else
	core::kernel::CPULauncher launcher;
#endif

	// Go through voxels
	launcher.LaunchGeneralKernel(
			n_voxels,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {

				// Natural index (0, N) ->
				//                        (workload_block_idx, voxel_index_in_block)
				int64_t workload_block_idx = workload_idx / resolution3;
				int64_t block_index = indices_ptr[workload_block_idx];
				int64_t voxel_index_in_block = workload_idx % resolution3;

				// block_index -> (x_block, y_block, z_block)
				int* block_key_ptr =
						block_keys_indexer.GetDataPtrFromCoord<int>(block_index);
				int64_t x_block = static_cast<int64_t>(block_key_ptr[0]);
				int64_t y_block = static_cast<int64_t>(block_key_ptr[1]);
				int64_t z_block = static_cast<int64_t>(block_key_ptr[2]);

				// voxel_idx -> (x_voxel, y_voxel, z_voxel)
				int64_t x_voxel, y_voxel, z_voxel;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block,
				                              &x_voxel, &y_voxel, &z_voxel);


				auto* voxel_center_pointer = voxel_centers_indexer.GetDataPtrFromCoord<float>(workload_idx);

				voxel_center_pointer[0] = static_cast<float>(x_block * block_resolution + x_voxel) * voxel_size;
				voxel_center_pointer[1] = static_cast<float>(y_block * block_resolution + y_voxel) * voxel_size;
				voxel_center_pointer[2] = static_cast<float>(z_block * block_resolution + z_voxel) * voxel_size;
			}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

#if defined(__CUDACC__)
void ExtractTSDFValuesAndWeightsCUDA
#else

void ExtractTSDFValuesAndWeightsCPU
#endif
		(const core::Tensor& indices,
		 const core::Tensor& block_values,
		 core::Tensor& voxel_values,
		 int64_t block_resolution) {

	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});

	// Output
#if defined(__CUDACC__)
	core::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(indices.GetLength());


	int64_t n_voxels = n_blocks * block_resolution3;
	// each voxel output will need a TSDF value and a weight value: n_voxels x 2
	voxel_values = core::Tensor::Zeros({n_voxels, 2}, core::Dtype::Float32,
	                                   block_values.GetDevice());

	// Real data indexers
	NDArrayIndexer voxel_values_indexer(voxel_values, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

	// Plain arrays that does not require indexers
	const auto* indices_ptr = indices.GetDataPtr<int64_t>();


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
			n_voxels, [=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
				// Natural index (0, N) ->
				//                        (workload_block_idx, voxel_index_in_block)
				int64_t block_idx = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;

				// voxel_idx -> (x_voxel, y_voxel, z_voxel)
				int64_t x_local, y_local, z_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block,
				                              &x_local, &y_local, &z_local);

				auto voxel_ptr = voxel_block_buffer_indexer
						.GetDataPtrFromCoord<voxel_t>(x_local, y_local, z_local, block_idx);

				auto voxel_value_pointer = voxel_values_indexer.GetDataPtrFromCoord<float>(workload_idx);

				voxel_value_pointer[0] = voxel_ptr->GetTSDF();
				voxel_value_pointer[1] = static_cast<float>(voxel_ptr->GetWeight());

			} // end lambda
				);
			}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}


#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ExtractValuesInExtentCUDA(
#else

void ExtractValuesInExtentCPU(
#endif
		int64_t min_x, int64_t min_y, int64_t min_z,
		int64_t max_x, int64_t max_y, int64_t max_z,
		const core::Tensor& block_indices,
		const core::Tensor& block_keys,
		const core::Tensor& block_values,
		core::Tensor& voxel_values,
		int64_t block_resolution) {

	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});

	// Output
#if defined(__CUDACC__)
	core::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(block_indices.GetLength());

	int64_t output_range_x = max_x - min_x;
	int64_t output_range_y = max_y - min_y;
	int64_t output_range_z = max_z - min_z;

	int64_t n_voxels = n_blocks * block_resolution3;
	// each voxel center will need three coordinates: n_voxels x 3
	voxel_values = core::Tensor::Ones({output_range_x, output_range_y, output_range_z},
	                                  core::Dtype::Float32, block_keys.GetDevice());
	voxel_values *= -2.0f;

	// Real data indexers
	NDArrayIndexer voxel_value_indexer(voxel_values, 3);
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

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
						n_voxels, [=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
				// Natural index (0, N) ->
				//                    (workload_block_idx, voxel_index_in_block)
				int64_t block_index = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;

				// block_index -> (x_block, y_block, z_block)
				int* block_key_ptr =
						block_keys_indexer.GetDataPtrFromCoord<int>(block_index);
				auto x_block = static_cast<int64_t>(block_key_ptr[0]);
				auto y_block = static_cast<int64_t>(block_key_ptr[1]);
				auto z_block = static_cast<int64_t>(block_key_ptr[2]);

				// voxel_idx -> (x_voxel, y_voxel, z_voxel)
				int64_t x_voxel_local, y_voxel_local, z_voxel_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block, &x_voxel_local, &y_voxel_local, &z_voxel_local);

				// at this point, (x_voxel, y_voxel, z_voxel) hold local
				// in-block coordinates. Compute the global voxel coordinates:
				int64_t x_voxel_global = x_block * block_resolution + x_voxel_local;
				int64_t y_voxel_global = y_block * block_resolution + y_voxel_local;
				int64_t z_voxel_global = z_block * block_resolution + z_voxel_local;

				int64_t x_voxel_out = x_voxel_global - min_x;
				int64_t y_voxel_out = y_voxel_global - min_y;
				int64_t z_voxel_out = z_voxel_global - min_z;

				if (x_voxel_out >= 0 && x_voxel_out < output_range_x &&
				    y_voxel_out >= 0 && y_voxel_out < output_range_y &&
				    z_voxel_out >= 0 && z_voxel_out < output_range_z) {
					auto* voxel_value_pointer =
							voxel_value_indexer.GetDataPtrFromCoord<float>(
									x_voxel_out, y_voxel_out, z_voxel_out);

					auto voxel_pointer = voxel_block_buffer_indexer.GetDataPtrFromCoord<voxel_t>(
							x_voxel_local, y_voxel_local, z_voxel_local, block_index);

					auto weight = voxel_pointer->GetWeight();

					if (weight > 0) {
						*voxel_value_pointer = voxel_pointer->GetTSDF();
					}
				}
			} // end element_kernel
				);
			}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

template<typename TLambdaRetrieveNode>
NNRT_CPU_OR_CUDA_DEVICE
void FindKNNAnchorsBruteForce(int* anchor_indices, float* squared_distances, const int anchor_count,
                              const int n_nodes, const Eigen::Vector3f& voxel_global_metric,
                              TLambdaRetrieveNode&& retrieve_node) {
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		squared_distances[i_anchor] = INFINITY;
	}
	int max_at_index = 0;
	float max_squared_distance = INFINITY;
	for (int i_node = 0; i_node < n_nodes; i_node++) {
		Eigen::Vector3f node = retrieve_node(i_node);
		float squared_distance = (node - voxel_global_metric).squaredNorm();

		if (squared_distance < max_squared_distance) {
			squared_distances[max_at_index] = squared_distance;
			anchor_indices[max_at_index] = i_node;

			//update the maximum distance within current anchor nodes
			max_at_index = 0;
			max_squared_distance = squared_distances[max_at_index];
			for (int i_anchor = 1; i_anchor < anchor_count; i_anchor++) {
				if (squared_distances[i_anchor] > max_squared_distance) {
					max_at_index = i_anchor;
					max_squared_distance = squared_distances[i_anchor];
				}
			}
		}
	}
}


template<typename TApplyBlendWarp>
void IntegrateWarped_Generic(
		const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
		core::Tensor& cos_voxel_ray_to_normal,
		int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
		const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
		const core::Tensor& intrinsics, const core::Tensor& extrinsics,
		const core::Tensor& warp_graph_nodes, const float node_coverage, const int anchor_count, const float depth_scale, const float depth_max,
		TApplyBlendWarp&& blend_function
) {
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

	// Output
#if defined(__CUDACC__)
	core::CUDACachedMemoryManager::ReleaseCache();
#endif

	int n_blocks = static_cast<int>(block_indices.GetLength());
	int64_t n_nodes = warp_graph_nodes.GetLength();

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
				// region ===================== FIND ANCHOR POINTS ================================
				int anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];
				FindKNNAnchorsBruteForce(anchor_indices, anchor_weights, anchor_count,
				                         n_nodes, voxel_global_metric,
				                         [&node_indexer](const int i_node) {
					                         auto node_pointer = node_indexer.GetDataPtrFromCoord<float>(i_node);
					                         Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
					                         return node;
				                         }
				);
				// endregion
				// region ===================== COMPUTE ANCHOR WEIGHTS ================================
				float weight_sum = 0.0;
				int valid_anchor_count = 0;
				for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
					float squared_distance = anchor_weights[i_anchor];
					// note: equivalent to distance > 2 * node_coverage, avoids sqrtf
					if (squared_distance > 4 * node_coverage_squared) {
						anchor_indices[i_anchor] = -1;
						continue;
					}
					float weight = expf(-squared_distance / (2 * node_coverage_squared));
					weight_sum += weight;
					anchor_weights[i_anchor] = weight;
					valid_anchor_count++;
				}
				if (valid_anchor_count < MINIMUM_VALID_ANCHOR_COUNT) {
					// TODO: verify
					//  a minimum of 1 node for fusion recommended by Neural Non-Rigid Tracking authors (?)
					return;
				}
				if (weight_sum > 0.0f) {
					for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
						anchor_weights[i_anchor] /= weight_sum;
					}
				} else if (anchor_count > 0) {
					for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
						anchor_weights[i_anchor] = 1.0f / anchor_count;
					}
				}
				// endregion
				// region ===================== CONVERT VOXEL TO CAMERA SPACE, WARP IT, AND PROJECT TO IMAGE ============================

				// voxel world coordinate (in voxels) -> voxel camera coordinate (in meters)
				float x_voxel_camera, y_voxel_camera, z_voxel_camera;
				transform_indexer.RigidTransform(voxel_global_metric.x(), voxel_global_metric.y(), voxel_global_metric.z(),
				                                 &x_voxel_camera, &y_voxel_camera, &z_voxel_camera);
				Eigen::Vector3f voxel_camera(x_voxel_camera, y_voxel_camera, z_voxel_camera);


				Eigen::Vector3f warped_voxel = blend_function(voxel_camera, anchor_indices, anchor_weights, node_indexer);
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

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void IntegrateWarpedDQ_CUDA(
#else
void IntegrateWarpedDQ_CPU(
#endif
		const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
		core::Tensor& cos_voxel_ray_to_normal,
		int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
		const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
		const core::Tensor& intrinsics, const core::Tensor& extrinsics,
		const core::Tensor& warp_graph_nodes, const core::Tensor& node_dual_quaternion_transformations,
		float node_coverage, int anchor_count, float depth_scale, float depth_max
) {

	NDArrayIndexer node_transform_indexer(node_dual_quaternion_transformations, 1);

	IntegrateWarped_Generic(
			block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_coverage, anchor_count, depth_scale, depth_max,
			[=] NNRT_CPU_OR_CUDA_DEVICE
					(const Eigen::Vector3f& voxel_camera, const int* anchor_indices, const float* anchor_weights,
					 const NDArrayIndexer& node_indexer) {
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


#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void IntegrateWarpedMatCUDA(
#else
void IntegrateWarpedMatCPU(
#endif
		const core::Tensor& block_indices, const core::Tensor& block_keys, core::Tensor& block_values,
		core::Tensor& cos_voxel_ray_to_normal,
		int64_t block_resolution, float voxel_size, float sdf_truncation_distance,
		const core::Tensor& depth_tensor, const core::Tensor& color_tensor, const core::Tensor& depth_normals,
		const core::Tensor& intrinsics, const core::Tensor& extrinsics,
		const core::Tensor& warp_graph_nodes, const core::Tensor& node_rotations, const core::Tensor& node_translations,
		const float node_coverage, const int anchor_count, const float depth_scale, const float depth_max
) {
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);

	IntegrateWarped_Generic(
			block_indices, block_keys, block_values, cos_voxel_ray_to_normal, block_resolution, voxel_size, sdf_truncation_distance,
			depth_tensor, color_tensor, depth_normals, intrinsics, extrinsics, warp_graph_nodes, node_coverage, anchor_count, depth_scale, depth_max,
			[=] NNRT_CPU_OR_CUDA_DEVICE
					(const Eigen::Vector3f& voxel_camera, const int* anchor_indices, const float* anchor_weights,
					 const NDArrayIndexer& node_indexer) {
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

} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt
