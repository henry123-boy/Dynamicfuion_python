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
#include <cfloat>

#include <Eigen/Geometry>

#include <open3d/core/Tensor.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/t/geometry/kernel/TSDFVoxel.h>
#include <open3d/t/geometry/kernel/TSDFVoxelGrid.h>

#include "core/PlatformIndependence.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "geometry/kernel/WarpableTSDFVoxelGrid.h"
#include "geometry/kernel/Defines.h"
#include "geometry/kernel/WarpUtilities.h"

#include "geometry/kernel/Segment.h"
#include "core/PlatformIndependentAtomics.h"

#ifndef __CUDACC__

#include <tbb/concurrent_unordered_set.h>

#endif


using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;
namespace kdtree = nnrt::core::kernel::kdtree;

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

	auto* kd_tree_nodes = warp_field.GetIndex().GetNodes();
	const int kd_tree_node_count = static_cast<int>(warp_field.GetIndex().GetNodeCount());

	//  Go through voxels
//@formatter:off
	DISPATCH_BYTESIZE_TO_VOXEL(
			voxel_block_buffer_indexer.ElementByteSize(),
			[&]() {
				open3d::core::ParallelFor(
						depth_tensor.GetDevice(), n_voxels,
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
				Eigen::Vector3f voxel_camera(x_voxel_camera, y_voxel_camera, z_voxel_camera);
				// endregion
				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];
				if (!warp::FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
						voxel_camera, node_coverage_squared
				)) {
					return;
				}
				// endregion
				// region ===================== WARP CAMERA-SPACE VOXEL AND PROJECT TO IMAGE ============================
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

template<open3d::core::Device::DeviceType TDeviceType>
void GetBoundingBoxesOfWarpedBlocks(open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_keys,
                                    const GraphWarpField& warp_field, float voxel_size, int64_t block_resolution,
                                    const open3d::core::Tensor& extrinsics) {
	//TODO: optionally, filter out voxel blocks (this is an unnecessary optimization unless we need to use a great multitude of voxel blocks)
	int64_t block_count = block_keys.GetLength();
	o3c::Device device = block_keys.GetDevice();
	bounding_boxes = o3c::Tensor({block_count, 6}, o3c::Float32, device);
	NDArrayIndexer bounding_box_indexer(bounding_boxes, 1);

	NDArrayIndexer block_key_indexer(block_keys, 1);
	TransformIndexer transform_indexer(o3c::Tensor::Eye(3, o3c::Float64, o3c::Device("CPU:0")), extrinsics, 1.0);


	float block_side_length = static_cast<float>(block_resolution) * voxel_size;

	auto* kd_tree_nodes = warp_field.GetIndex().GetNodes();
	const int kd_tree_node_count = static_cast<int>(warp_field.GetIndex().GetNodeCount());
	NDArrayIndexer node_indexer(warp_field.nodes, 1);
	int anchor_count = warp_field.anchor_count;
	float node_coverage_squared = warp_field.node_coverage * warp_field.node_coverage;
	NDArrayIndexer node_rotation_indexer(warp_field.rotations, 1);
	NDArrayIndexer node_translation_indexer(warp_field.translations, 1);

	open3d::core::ParallelFor(
			device, block_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto* block_key_ptr = block_key_indexer.GetDataPtr<int32_t>(workload_idx);

				auto block_x0 = static_cast<float>(block_key_ptr[0]);
				auto block_y0 = static_cast<float>(block_key_ptr[1]);
				auto block_z0 = static_cast<float>(block_key_ptr[2]);

				//TODO: abstract away generation of coordinates in a separate function that just fills a static array of Vector3f elements
				auto block_x1 = block_x0 + block_side_length;
				auto block_y1 = block_y0 + block_side_length;
				auto block_z1 = block_z0 + block_side_length;

				Eigen::Vector3f block_corner_000(block_x0, block_y0, block_z0);
				Eigen::Vector3f block_corner_001(block_x0, block_y0, block_z1);
				Eigen::Vector3f block_corner_010(block_x0, block_y1, block_z0);
				Eigen::Vector3f block_corner_100(block_x1, block_y0, block_z0);
				Eigen::Vector3f block_corner_011(block_x0, block_y1, block_z1);
				Eigen::Vector3f block_corner_101(block_x1, block_y0, block_z1);
				Eigen::Vector3f block_corner_110(block_x1, block_y1, block_z0);
				Eigen::Vector3f block_corner_111(block_x1, block_y1, block_z1);

				Eigen::Vector3f block_corners[] = {block_corner_000, block_corner_001, block_corner_010, block_corner_100,
				                                   block_corner_011, block_corner_101, block_corner_110, block_corner_111};
				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];

				auto* bounding_box_min_ptr = bounding_box_indexer.GetDataPtr<float>(workload_idx);

				Eigen::Map<Eigen::Vector3f> box_min(bounding_box_min_ptr);
				box_min.x() = FLT_MAX;
				box_min.y() = FLT_MAX;
				box_min.z() = FLT_MAX;

				Eigen::Map<Eigen::Vector3f> box_max(bounding_box_min_ptr + 3);

				box_max.x() = -FLT_MAX;
				box_max.y() = -FLT_MAX;
				box_max.z() = -FLT_MAX;


				for (auto& corner: block_corners) {
					Eigen::Vector3f corner_camera;
					transform_indexer.RigidTransform(corner.x(), corner.y(), corner.z(),
					                                 corner_camera.data(), corner_camera.data() + 1, corner_camera.data() + 2);
					warp::FindAnchorsAndWeightsForPointEuclidean_KDTree<TDeviceType>(
							anchor_indices, anchor_weights, anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
							corner_camera, node_coverage_squared
					);
					Eigen::Vector3f warped_corner(0.f, 0.f, 0.f);
					warp::BlendWarp(warped_corner, anchor_indices, anchor_weights, anchor_count, node_indexer,
					                node_rotation_indexer, node_translation_indexer, corner_camera);
					if (box_min.x() > warped_corner.x()) box_min.x() = warped_corner.x();
					else if (box_max.x() < warped_corner.x()) box_max.x() = warped_corner.x();
					if (box_min.y() > warped_corner.y()) box_min.y() = warped_corner.y();
					else if (box_max.y() < warped_corner.y()) box_max.y() = warped_corner.y();
					if (box_min.z() > warped_corner.z()) box_min.z() = warped_corner.z();
					else if (box_max.z() < warped_corner.z()) box_max.z() = warped_corner.z();
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void GetAxisAlignedBoxesInterceptingSurfaceMask(open3d::core::Tensor& mask, const open3d::core::Tensor& boxes, const open3d::core::Tensor& intrinsics,
                                                const open3d::core::Tensor& depth, float depth_scale, float depth_max, int32_t stride,
                                                float truncation_distance) {
	o3c::Device device = boxes.GetDevice();
	int64_t box_count = boxes.GetLength();
	mask = o3c::Tensor::Zeros({box_count}, o3c::Bool, device);

	TransformIndexer transform_indexer(intrinsics, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0);

	NDArrayIndexer mask_indexer(mask, 1);
	NDArrayIndexer box_indexer(boxes, 1);
	NDArrayIndexer depth_indexer(depth, 2);

	auto rows_strided = depth_indexer.GetShape(0) / stride;
	auto cols_strided = depth_indexer.GetShape(1) / stride;
	int64_t sampled_pixel_count = rows_strided * cols_strided;

	o3c::Blob segments(static_cast<int64_t>(sizeof(Segment)) * sampled_pixel_count, device);
	auto* segment_data = reinterpret_cast<Segment*>(segments.GetDataPtr());
	int64_t segment_count;
	//@formatter:off
	DISPATCH_DTYPE_TO_TEMPLATE(
			depth.GetDtype(),
			[&]() {
				NNRT_DECLARE_ATOMIC(uint32_t , segment_count_atomic);
				NNRT_INITIALIZE_ATOMIC(uint32_t, segment_count_atomic, 0);

				o3c::ParallelFor( device, sampled_pixel_count,
								  NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int v = (workload_idx / cols_strided) * stride;
				int u = (workload_idx % cols_strided) * stride;
				float depth = *depth_indexer.GetDataPtr<scalar_t>(u, v) / depth_scale;
				if (depth > 0 && depth < depth_max) {
					Eigen::Vector3f segment_start, segment_end;
					float segment_start_depth = depth - truncation_distance;
					float segment_end_depth = depth + truncation_distance;
					transform_indexer.Unproject(u, v, segment_start_depth, segment_start.data(), segment_start.data() + 1,
					                            segment_start.data() + 2);
					transform_indexer.Unproject(u, v, segment_end_depth, segment_end.data(), segment_end.data() + 1,
					                            segment_end.data() + 2);
					uint32_t segment_index = NNRT_ATOMIC_ADD(segment_count_atomic, (uint32_t)1);
					segment_data[segment_index] = Segment(segment_start, segment_end);
				}

			});
			segment_count = NNRT_GET_ATOMIC_VALUE_CPU(segment_count_atomic);
			NNRT_CLEAN_UP_ATOMIC(segment_count_atomic);
	});
	//@formatter:on

	int64_t intersection_check_count = segment_count * box_count;

	o3c::ParallelFor(
			device, intersection_check_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				int64_t i_segment = workload_idx % segment_count;
				int64_t i_box = workload_idx / segment_count;
				auto box_data = box_indexer.template GetDataPtr<float>(i_box);
				Eigen::Map<Eigen::Vector3f> box_min(box_data);
				Eigen::Map<Eigen::Vector3f> box_max(box_data + 3);

				if (segment_data[i_segment].IntersectsAxisAlignedBox(box_min, box_max)) {
					*mask_indexer.GetDataPtr<bool>(i_box) = true;
				}
			}

	);
}

} // namespace nnrt::geometry::kernel::tsdf
