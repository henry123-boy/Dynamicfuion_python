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

#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentAtomics.h"
#include "geometry/kernel/NonRigidSurfaceVoxelBlockGrid.h"
#include "geometry/kernel/VoxelGridDtypeDispatch.h"
#include "geometry/kernel/Defines.h"
#include "geometry/kernel/Segment.h"


#ifndef __CUDACC__

#include <tbb/concurrent_unordered_set.h>

#endif


using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
namespace kdtree = nnrt::core::kernel::kdtree;

namespace nnrt::geometry::kernel::voxel_grid {


using ArrayIndexer = TArrayIndexer<index_t>;

template<open3d::core::Device::DeviceType TDeviceType, typename input_depth_t, typename input_color_t, typename weight_t, typename color_t>
void
IntegrateNonRigid_Generic
		(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys,
		 open3d::t::geometry::TensorMap& block_value_map, open3d::core::Tensor& cos_voxel_ray_to_normal,
		 index_t block_resolution, float voxel_size, float sdf_truncation_distance,
		 const open3d::core::Tensor& depth, const open3d::core::Tensor& color,
		 const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& depth_intrinsics, const open3d::core::Tensor& color_intrinsics,
		 const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field, float depth_scale, float depth_max) {
	using tsdf_t = float;
	index_t block_resolution_cubed = block_resolution * block_resolution * block_resolution;
	// Shape / transform indexers, no data involved
	ArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});
	TransformIndexer depth_transform_indexer(depth_intrinsics, extrinsics, 1.0);
	TransformIndexer color_transform_indexer(color_intrinsics, o3c::Tensor::Eye(4, o3c::Dtype::Float64, o3c::Device("CPU:0")));

	int block_count = static_cast<int>(block_indices.GetLength());
	int64_t voxel_count = block_count * block_resolution_cubed;

	// cosine value for each pixel
	cos_voxel_ray_to_normal = o3c::Tensor::Zeros(depth.GetShape(), o3c::Dtype::Float32, block_keys.GetDevice());

	// Data structure indexers
	ArrayIndexer block_keys_indexer(block_keys, 1);

	// Image indexers
	ArrayIndexer depth_indexer(depth, 2);
	ArrayIndexer cosine_indexer(cos_voxel_ray_to_normal, 2);
	ArrayIndexer normals_indexer(depth_normals, 2);

	// Optional color integration
	ArrayIndexer color_indexer;
	bool integrate_color = false;
	if (color.NumElements() != 0 && block_value_map.Contains("color")) {
		integrate_color = true;
	}

	// Plain array that does not require indexers
	const auto* indices_ptr = block_indices.GetDataPtr<int64_t>();
	color_t* color_base_ptr = nullptr;
	float color_multiplier = 1.0;
	if (integrate_color) {
		color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
		color_indexer = ArrayIndexer(color, 2);

		// Float32: [0, 1] -> [0, 255]
		if (color.GetDtype() == o3c::Float32) {
			color_multiplier = 255.0;
		}
	}
	auto* tsdf_base_ptr = block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
	auto* weight_base_ptr = block_value_map.at("weight").GetDataPtr<weight_t>();

//@formatter:off
	open3d::core::ParallelFor(
			depth.GetDevice(), voxel_count,
			[=] OPEN3D_DEVICE (int64_t workload_idx) {
//@formatter:on
				// region ===================== COMPUTE VOXEL COORDINATE & CAMERA COORDINATE ================================
				// Natural index (0, N) ->
				//                    (workload_block_idx, voxel_index_in_block)
				index_t block_index = indices_ptr[workload_idx / block_resolution_cubed];
				index_t voxel_index_in_block = workload_idx % block_resolution_cubed;


				// block_index -> x_block, y_block, z_block (in voxel hash blocks)
				auto* block_key_ptr =
						block_keys_indexer.GetDataPtr<index_t>(block_index);
				index_t x_block = block_key_ptr[0];
				index_t y_block = block_key_ptr[1];
				index_t z_block = block_key_ptr[2];

				// voxel_idx -> x_voxel_local, y_voxel_local, z_voxreel_local (in voxels)
				index_t x_voxel_local, y_voxel_local, z_voxel_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block, &x_voxel_local, &y_voxel_local, &z_voxel_local);

				// at this point, (x_voxel, y_voxel, z_voxel) hold local
				// in-block coordinates. Compute the global voxel coordinates (in voxels, then meters)
				Eigen::Vector3i voxel_global(x_block * block_resolution + x_voxel_local,
				                             y_block * block_resolution + y_voxel_local,
				                             z_block * block_resolution + z_voxel_local);
				Eigen::Vector3f voxel_global_metric = voxel_global.cast<float>() * voxel_size;

				// voxel world coordinate (in voxels) -> voxel camera coordinate (in meters)
				float x_voxel_camera, y_voxel_camera, z_voxel_camera;
				depth_transform_indexer.RigidTransform(voxel_global_metric.x(), voxel_global_metric.y(), voxel_global_metric.z(),
				                                       &x_voxel_camera, &y_voxel_camera, &z_voxel_camera);
				Eigen::Vector3f voxel_camera(x_voxel_camera, y_voxel_camera, z_voxel_camera);
				// endregion
				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];

				if(!warp_field.template ComputeAnchorsForPoint<TDeviceType, true>(anchor_indices, anchor_weights, voxel_camera)){
					return;
				}
				// endregion
				// region ===================== WARP CAMERA-SPACE VOXEL AND PROJECT TO IMAGE ============================
				auto warped_voxel = warp_field.template WarpPoint<TDeviceType>(voxel_camera, anchor_indices, anchor_weights);

				if (warped_voxel.z() < 0) {
					// voxel is behind camera
					return;
				}

				// coordinate in image (in pixels)
				float u_precise, v_precise;
				depth_transform_indexer.Project(warped_voxel.x(), warped_voxel.y(), warped_voxel.z(), &u_precise, &v_precise);

				if (!depth_indexer.InBoundary(u_precise, v_precise)) {
					return;
				}

				// endregion
				// region ===================== SAMPLE IMAGES AND COMPUTE THE ACTUAL TSDF & COLOR UPDATE =======================
				auto u_rounded = static_cast<index_t>(roundf(u_precise));
				auto v_rounded = static_cast<index_t>(roundf(v_precise));

				float depth = (*depth_indexer.GetDataPtr<input_depth_t>(u_rounded, v_rounded)) / depth_scale;
				if (depth <= 0.0f || depth > depth_max) {
					return;
				}

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
				if (psdf <= -sdf_truncation_distance || cosine > 0.5f) {
					return;
				}
				index_t linear_voxel_index = block_index * block_resolution_cubed + voxel_index_in_block;

				float tsdf_normalized =
						(psdf < sdf_truncation_distance ? psdf : sdf_truncation_distance) / sdf_truncation_distance;
				tsdf_t* tsdf_ptr = tsdf_base_ptr + linear_voxel_index;
				weight_t* weight_ptr = weight_base_ptr + linear_voxel_index;

				float weight = *weight_ptr;
				float inverted_weight_sum = 1.0f / (weight + 1);
				*tsdf_ptr = (weight * (*tsdf_ptr) + tsdf_normalized) * inverted_weight_sum;

				if (integrate_color) {
					color_t* color_ptr = color_base_ptr + 3 * linear_voxel_index;

					// Unproject ui, vi with depth_intrinsics, then project back with color_intrinsics
					float x, y, z;
					depth_transform_indexer.Unproject(static_cast<float>(u_rounded), static_cast<float>(v_rounded), 1.0, &x, &y, &z);
					color_transform_indexer.Project(x, y, z, &u_precise, &v_precise);

					if (color_indexer.InBoundary(u_precise, v_precise)) {
						u_rounded = static_cast<index_t>(roundf(u_precise));
						v_rounded = static_cast<index_t>(roundf(v_precise));

						auto* input_color_ptr = color_indexer.GetDataPtr<input_color_t>(u_rounded, v_rounded);

						for (index_t i_channel = 0; i_channel < 3; i_channel++) {
							color_ptr[i_channel] =
									(weight * color_ptr[i_channel] + input_color_ptr[i_channel] * color_multiplier) * inverted_weight_sum;
						}
					}
				}

				// endregion
			} /* end element_kernel */ ); // end ParallelFor call
}

template<open3d::core::Device::DeviceType TDeviceType>
void
IntegrateNonRigid(
		const open3d::core::Tensor& block_indices, const open3d::core::Tensor& block_keys,
		open3d::t::geometry::TensorMap& block_value_map, open3d::core::Tensor& cos_voxel_ray_to_normal,
		index_t block_resolution, float voxel_size, float sdf_truncation_distance,
		const open3d::core::Tensor& depth, const open3d::core::Tensor& color,
		const open3d::core::Tensor& depth_normals, const open3d::core::Tensor& depth_intrinsics, const open3d::core::Tensor& color_intrinsics,
		const open3d::core::Tensor& extrinsics, const GraphWarpField& warp_field, float depth_scale, float depth_max
) {

	o3c::Dtype block_weight_dtype = o3c::Dtype::Float32;
	o3c::Dtype block_color_dtype = o3c::Dtype::Float32;

	if (block_value_map.Contains("weight")) {
		block_weight_dtype = block_value_map.at("weight").GetDtype();
	}
	if (block_value_map.Contains("color")) {
		block_color_dtype = block_value_map.at("color").GetDtype();
	}

	o3c::Dtype input_depth_dtype = depth.GetDtype();
	o3c::Dtype input_color_dtype = (input_depth_dtype == o3c::Dtype::Float32)
	                               ? o3c::Dtype::Float32
	                               : o3c::Dtype::UInt8;



	//  Go through voxels
//@formatter:off
	DISPATCH_INPUT_DTYPE_TO_TEMPLATE(
			input_depth_dtype, input_color_dtype, [&] {
		DISPATCH_VALUE_DTYPE_TO_TEMPLATE(block_weight_dtype, block_color_dtype, [&]() {
//@formatter:on
//TODO: when CUDA finally starts supporting C++20, revise this to use a templated lambda (so you don't have to maintain a rewrite of all the parameters
// and the super-long call). See https://stackoverflow.com/a/62932369/844728 for reference.
			IntegrateNonRigid_Generic<TDeviceType, input_depth_t, input_color_t, weight_t, color_t>(
					block_indices, block_keys, block_value_map, cos_voxel_ray_to_normal, block_resolution, voxel_size,
					sdf_truncation_distance, depth, color, depth_normals, depth_intrinsics, color_intrinsics, extrinsics,
					warp_field, depth_scale, depth_max);
		} /* end lambda */ ); // end DISPATCH_VALUE_DTYPE_TO_TEMPLATE macro call
	} /* end lambda  */ ); // end DISPATCH_INPUT_DTYPE_TO_TEMPLATE macro call
#if defined(__CUDACC__)
	o3c::cuda::Synchronize();
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void GetBoundingBoxesOfWarpedBlocks(open3d::core::Tensor& bounding_boxes, const open3d::core::Tensor& block_keys,
                                    const GraphWarpField& warp_field, float voxel_size, index_t block_resolution,
                                    const open3d::core::Tensor& extrinsics) {
	//TODO: optionally, filter out voxel blocks (this is an unnecessary optimization unless we need to use a great multitude of voxel blocks)
	int64_t block_count = block_keys.GetLength();
	o3c::Device device = block_keys.GetDevice();
	bounding_boxes = o3c::Tensor({block_count, 6}, o3c::Float32, device);
	NDArrayIndexer bounding_box_indexer(bounding_boxes, 1);

	NDArrayIndexer block_key_indexer(block_keys, 1);
	TransformIndexer transform_indexer(o3c::Tensor::Eye(3, o3c::Float64, o3c::Device("CPU:0")), extrinsics, 1.0);

	float block_side_length = static_cast<float>(block_resolution) * voxel_size;

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
					warp_field.template ComputeAnchorsForPoint<TDeviceType, true>(anchor_indices, anchor_weights, corner_camera);
					Eigen::Vector3f warped_corner = warp_field.template WarpPoint<TDeviceType>(corner_camera, anchor_indices, anchor_weights);
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

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractVoxelValuesAndCoordinates(o3c::Tensor& voxel_values_and_coordinates, const open3d::core::Tensor& block_indices,
                                      const open3d::core::Tensor& block_keys, const open3d::t::geometry::TensorMap& block_value_map,
                                      int64_t block_resolution, float voxel_size) {

	using tsdf_t = float;

	o3c::Dtype block_weight_dtype = o3c::Dtype::Float32;
	o3c::Dtype block_color_dtype = o3c::Dtype::Float32;

	bool has_weight = false;
	int64_t output_channel_count = 4;
	if (block_value_map.Contains("weight")) {
		has_weight = true;
		block_weight_dtype = block_value_map.at("weight").GetDtype();
		output_channel_count += 1;
	}
	bool has_color = false;
	if (block_value_map.Contains("color")) {
		has_color = true;
		block_color_dtype = block_value_map.at("color").GetDtype();
		output_channel_count += 3;
	}

	int64_t block_resolution_cubed = block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	ArrayIndexer voxel_indexer({block_resolution, block_resolution, block_resolution});

	int block_count = static_cast<int>(block_indices.GetLength());


	int64_t voxel_count = block_count * block_resolution_cubed;
	// In a fully-colored TSDF voxel grid, each voxel output will need a 3d coordinate, TSDF value, weight value, and 3 color channels,
	// all in float form: voxel_count x (3 + 1 + 1 + 3)
	voxel_values_and_coordinates = o3c::Tensor::Zeros({voxel_count, output_channel_count}, o3c::Dtype::Float32, block_indices.GetDevice());

	// Indexer for block coordinates
	ArrayIndexer block_keys_indexer(block_keys, 1);
	// Output data indexer
	NDArrayIndexer voxel_values_indexer(voxel_values_and_coordinates, 1);

	// Plain arrays that does not require indexers
	const auto* indices_ptr = block_indices.GetDataPtr<int64_t>();


	//  Go through voxels
	DISPATCH_VALUE_DTYPE_TO_TEMPLATE(block_weight_dtype, block_color_dtype, [&]() {
		const color_t* color_base_ptr = nullptr;
		if (has_color) {
			color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
		}
		const weight_t* weight_base_ptr = nullptr;
		if (has_weight) {
			weight_base_ptr = block_value_map.at("weight").GetDataPtr<weight_t>();
		}
		const auto* tsdf_base_ptr = block_value_map.at("tsdf").GetDataPtr<tsdf_t>();

		open3d::core::ParallelFor(
				block_indices.GetDevice(), voxel_count,
//@formatter:off
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
		// Natural index (0, N) -> (workload_block_idx, voxel_index_in_block)
		index_t block_index = indices_ptr[workload_idx / block_resolution_cubed];
		index_t voxel_index_in_block = workload_idx % block_resolution_cubed;


		// block_index -> x_block, y_block, z_block (in voxel hash blocks)
		auto* block_key_ptr = block_keys_indexer.GetDataPtr<index_t>(block_index);
		index_t x_block = block_key_ptr[0];
		index_t y_block = block_key_ptr[1];
		index_t z_block = block_key_ptr[2];

		// voxel_idx -> x_voxel_local, y_voxel_local, z_voxel_local (in voxels)
		index_t x_voxel_local, y_voxel_local, z_voxel_local;
		voxel_indexer.WorkloadToCoord(voxel_index_in_block, &x_voxel_local, &y_voxel_local, &z_voxel_local);

		Eigen::Vector3f voxel_global(x_block * block_resolution + x_voxel_local,
		                             y_block * block_resolution + y_voxel_local,
		                             z_block * block_resolution + z_voxel_local);
		Eigen::Vector3f voxel_global_metric = voxel_global * voxel_size;

		index_t linear_voxel_index = block_index * block_resolution_cubed + voxel_index_in_block;

		const auto* tsdf_pointer = tsdf_base_ptr + linear_voxel_index;
		auto voxel_value_pointer = voxel_values_indexer.GetDataPtr<float>(workload_idx);

		voxel_value_pointer[0] = voxel_global_metric.x();
		voxel_value_pointer[1] = voxel_global_metric.y();
		voxel_value_pointer[2] = voxel_global_metric.z();
		voxel_value_pointer[3] = (float) *tsdf_pointer;
		int index = 4;
		if (has_weight) {
			const auto* weight_pointer = weight_base_ptr + linear_voxel_index;
			voxel_value_pointer[index] = (float) *weight_pointer;
			index++;
		}
		if (has_color) {
			const auto* color_pointer = color_base_ptr + 3 * linear_voxel_index;
			for (int i_channel = 0; i_channel < 3; i_channel++, index++) {
				voxel_value_pointer[index] = (float) color_pointer[i_channel];
			}
		}

	} /* end lambda */ ); // end ParallelFor call
	} /* end lambda */ ); // end DISPATCH_VALUE_DTYPE_TO_TEMPLATE macro call
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractVoxelValuesAt(o3c::Tensor& voxel_values, const o3c::Tensor& query_coordinates, const open3d::core::Tensor& query_block_indices,
                          const open3d::core::Tensor& block_keys, const open3d::t::geometry::TensorMap& block_value_map,
                          int64_t block_resolution, float voxel_size) {

	using tsdf_t = float;

	o3c::Dtype block_weight_dtype = o3c::Dtype::Float32;
	o3c::Dtype block_color_dtype = o3c::Dtype::Float32;

	bool has_weight = false;
	int64_t output_channel_count = 4;
	if (block_value_map.Contains("weight")) {
		has_weight = true;
		block_weight_dtype = block_value_map.at("weight").GetDtype();
		output_channel_count += 1;
	}
	bool has_color = false;
	if (block_value_map.Contains("color")) {
		has_color = true;
		block_color_dtype = block_value_map.at("color").GetDtype();
		output_channel_count += 3;
	}

	int64_t block_resolution_cubed = block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	ArrayIndexer voxel_indexer({block_resolution, block_resolution, block_resolution});

	int64_t query_voxel_count = query_coordinates.GetLength();
	// In a fully-colored TSDF voxel grid, each voxel output will need a 3d coordinate, TSDF value, weight value, and 3 color channels,
	// all in float form: query_voxel_count x (3 + 1 + 1 + 3)
	voxel_values = o3c::Tensor::Ones({query_voxel_count, output_channel_count}, o3c::Dtype::Float32, query_block_indices.GetDevice()) * -2.f;

	// Indexer for block coordinate keys
	ArrayIndexer block_keys_indexer(block_keys, 1);
	// Indexer for query voxel coordinates
	NDArrayIndexer query_coordinate_indexer(query_coordinates, 1);
	// Output data indexer
	NDArrayIndexer voxel_values_indexer(voxel_values, 1);

	// Plain arrays that do not require indexers
	const auto* query_block_index_ptr = query_block_indices.GetDataPtr<int32_t>();


	//  Go through voxels
	DISPATCH_VALUE_DTYPE_TO_TEMPLATE(block_weight_dtype, block_color_dtype, [&]() {
		const color_t* color_base_ptr = nullptr;
		if (has_color) {
			color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
		}
		const weight_t* weight_base_ptr = nullptr;
		if (has_weight) {
			weight_base_ptr = block_value_map.at("weight").GetDataPtr<weight_t>();
		}
		const auto* tsdf_base_ptr = block_value_map.at("tsdf").GetDataPtr<tsdf_t>();

		open3d::core::ParallelFor(
				query_block_indices.GetDevice(), query_voxel_count,
//@formatter:off
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
		// Natural index (0, N) -> (workload_block_idx, voxel_index_in_block)
		index_t query_block_index = query_block_index_ptr[workload_idx];
		Eigen::Map<Eigen::Vector3i> query_voxel_global(query_coordinate_indexer.GetDataPtr<int32_t>(workload_idx));
		index_t voxel_index_in_block;
		voxel_indexer.CoordToWorkload(query_voxel_global.x(), query_voxel_global.y(), query_voxel_global.z(), &voxel_index_in_block);

		Eigen::Vector3f voxel_global_metric = query_voxel_global.cast<float>() * voxel_size;

		index_t linear_voxel_index = query_block_index * block_resolution_cubed + voxel_index_in_block;

		const auto* tsdf_pointer = tsdf_base_ptr + linear_voxel_index;
		auto voxel_value_pointer = voxel_values_indexer.GetDataPtr<float>(workload_idx);

		voxel_value_pointer[0] = voxel_global_metric.x();
		voxel_value_pointer[1] = voxel_global_metric.y();
		voxel_value_pointer[2] = voxel_global_metric.z();
		voxel_value_pointer[3] = (float) *tsdf_pointer;
		int index = 4;
		if (has_weight) {
			const auto* weight_pointer = weight_base_ptr + linear_voxel_index;
			voxel_value_pointer[index] = (float) *weight_pointer;
			index++;
		}
		if (has_color) {
			const auto* color_pointer = color_base_ptr + 3 * linear_voxel_index;
			for (int i_channel = 0; i_channel < 3; i_channel++, index++) {
				voxel_value_pointer[index] = (float) color_pointer[i_channel];
			}
		}

	} /* end lambda */ ); // end ParallelFor call
	} /* end lambda */ ); // end DISPATCH_VALUE_DTYPE_TO_TEMPLATE macro call
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

} // namespace nnrt::geometry::kernel::tsdf
