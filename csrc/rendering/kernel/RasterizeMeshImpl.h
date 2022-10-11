//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/6/22.
//  Copyright (c) 2022 Gregory Kramida
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
#pragma once

// 3rd party
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/utility/Logging.h>
#include <Eigen/Dense>

// local
#include "core/functional/kernel/BubbleSort.h"
#include "core/PlatformIndependentAtomics.h"
#include "rendering/kernel/RasterizeMesh.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/RayFaceIntersection.h"



namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;


namespace nnrt::rendering::kernel {

//TODO: test whether serial version is slower
#define NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractClippedFaceVerticesInNormalizedCameraSpace(open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
                                                       const open3d::core::Tensor& vertex_positions_camera,
                                                       const open3d::core::Tensor& triangle_vertex_indices,
                                                       const open3d::core::Tensor& normalized_camera_space_matrix,
                                                       kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
                                                       float near_clipping_distance,
                                                       float far_clipping_distance) {
	o3c::Device device = vertex_positions_camera.GetDevice();
	NNRT_DECLARE_ATOMIC_INT(unclipped_face_count);
	NNRT_INITIALIZE_ATOMIC(int, unclipped_face_count, 0);

	auto face_count = triangle_vertex_indices.GetLength();
	// input indexers
	o3tgk::NDArrayIndexer face_vertex_index_indexer(triangle_vertex_indices, 1);
	o3tgk::NDArrayIndexer vertex_position_indexer(vertex_positions_camera, 1);

	// output & output indexers
	open3d::core::Tensor vertex_positions_normalized_camera_raw({face_count, 3, 3}, o3c::Float32, device);
	o3tgk::NDArrayIndexer normalized_face_vertex_indexer(vertex_positions_normalized_camera_raw, 1);

	o3tgk::TransformIndexer perspective_transform(normalized_camera_space_matrix, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0f);


	o3c::ParallelFor(
			device, face_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				Eigen::Map<Eigen::Matrix<int64_t, 1, 3>> face_vertex_indices(face_vertex_index_indexer.GetDataPtr<int64_t>(workload_idx));

				Eigen::Map<Eigen::Vector3f> face_vertex0(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(0)));
				Eigen::Map<Eigen::Vector3f> face_vertex1(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(1)));
				Eigen::Map<Eigen::Vector3f> face_vertex2(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(2)));

#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Map<Eigen::Vector3f> face_vertices[] = {face_vertex0, face_vertex1, face_vertex2};
#endif
				// clip if all vertices are too near or too far
#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				bool have_vertex_within_clipping_range = false;
				for (auto& face_vertex: face_vertices) {
					have_vertex_within_clipping_range |= face_vertex.z() >= near_clipping_distance;
					have_vertex_within_clipping_range |= face_vertex.z() <= far_clipping_distance;
				}
				if (!have_vertex_within_clipping_range) {
					return;
				}
#else
				if ((face_vertex0.z() > far_clipping_distance &&
					 face_vertex1.z() > far_clipping_distance &&
					 face_vertex2.z() > far_clipping_distance) ||
					(face_vertex0.z() < near_clipping_distance &&
					 face_vertex1.z() < near_clipping_distance &&
					 face_vertex2.z() < near_clipping_distance)) {
					return;
				}
#endif

#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Vector2f normalized_face_vertices_xy[3];
				bool has_inliner_vertex = false;
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					perspective_transform.Project(face_vertices[i_vertex].x(), face_vertices[i_vertex].y(), face_vertices[i_vertex].z(),
					                              &normalized_face_vertices_xy[i_vertex].x(), &normalized_face_vertices_xy[i_vertex].y());
					has_inliner_vertex |= normalized_camera_space_xy_range.Contains(normalized_face_vertices_xy[i_vertex]);
				}
				if (!has_inliner_vertex) {
					return;
				}
#else
				Eigen::Vector2f normalized_face_vertex_xy0, normalized_face_vertex_xy1, normalized_face_vertex_xy2;

				perspective_transform.Project(face_vertex0.x(), face_vertex0.y(), face_vertex0.z(),
											  &normalized_face_vertex_xy0.x(), &normalized_face_vertex_xy0.y());
				perspective_transform.Project(face_vertex1.x(), face_vertex1.y(), face_vertex1.z(),
											  &normalized_face_vertex_xy1.x(), &normalized_face_vertex_xy1.y());
				perspective_transform.Project(face_vertex2.x(), face_vertex2.y(), face_vertex2.z(),
											  &normalized_face_vertex_xy2.x(), &normalized_face_vertex_xy2.y());

				if (!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy0) &&
					!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy1) &&
					!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy2)) {
					// face is outside of the view frustum's top/bottom/left/right boundaries
					return;
				}

#endif
				auto output_face_index = static_cast<int64_t>(NNRT_ATOMIC_ADD(unclipped_face_count, 1));
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex0(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index));
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex1(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 3);
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex2(normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 6);
#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Map<Eigen::Vector3f> normalized_face_vertices[] = {normalized_face_vertex0, normalized_face_vertex1, normalized_face_vertex2};
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					normalized_face_vertices[i_vertex].x() = normalized_face_vertices_xy[i_vertex].x();
					// important: flip the y-coordinate to reflect pixel space
					normalized_face_vertices[i_vertex].y() = normalized_face_vertices_xy[i_vertex].y();
					normalized_face_vertices[i_vertex].z() = face_vertices[i_vertex].z();
				}
#else
				normalized_face_vertex0.x() = normalized_face_vertex_xy0.x();
				normalized_face_vertex0.y() = normalized_face_vertex_xy0.y();
				normalized_face_vertex0.z() = face_vertex0.z();
				normalized_face_vertex1.x() = normalized_face_vertex_xy1.x();
				normalized_face_vertex1.y() = normalized_face_vertex_xy1.y();
				normalized_face_vertex1.z() = face_vertex1.z();
				normalized_face_vertex2.x() = normalized_face_vertex_xy2.x();
				normalized_face_vertex2.y() = normalized_face_vertex_xy2.y();
				normalized_face_vertex2.z() = face_vertex2.z();
#endif
			}
	);

	int unclipped_face_count_host = NNRT_GET_ATOMIC_VALUE_CPU(unclipped_face_count);
	vertex_positions_clipped_normalized_camera = vertex_positions_normalized_camera_raw.Slice(0, 0, unclipped_face_count_host);
}

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractFaceVerticesAndClippingMaskInNormalizedCameraSpace(
		open3d::core::Tensor& vertex_positions_normalized_camera,
		open3d::core::Tensor& clipped_face_mask,
		const open3d::core::Tensor& vertex_positions_camera,
		const open3d::core::Tensor& triangle_vertex_indices,
		const open3d::core::Tensor& normalized_camera_space_matrix,
		kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
		float near_clipping_distance,
		float far_clipping_distance
) {
	o3c::Device device = vertex_positions_camera.GetDevice();
	NNRT_DECLARE_ATOMIC_INT(unclipped_face_count);
	NNRT_INITIALIZE_ATOMIC(int, unclipped_face_count, 0);

	auto face_count = triangle_vertex_indices.GetLength();
	// input indexers
	o3tgk::NDArrayIndexer face_vertex_index_indexer(triangle_vertex_indices, 1);
	o3tgk::NDArrayIndexer vertex_position_indexer(vertex_positions_camera, 1);

	// output & output indexers
	vertex_positions_normalized_camera = open3d::core::Tensor({face_count, 3, 3}, o3c::Float32, device);
	o3tgk::NDArrayIndexer normalized_face_vertex_indexer(vertex_positions_normalized_camera, 1);
	clipped_face_mask = open3d::core::Tensor::Ones({face_count}, o3c::Bool, device);
	bool* clipped_face_mask_ptr = clipped_face_mask.template GetDataPtr<bool>();

	o3tgk::TransformIndexer perspective_transform(normalized_camera_space_matrix, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0f);


	o3c::ParallelFor(
			device, face_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				Eigen::Map<Eigen::Matrix<int64_t, 1, 3>> face_vertex_indices(face_vertex_index_indexer.GetDataPtr<int64_t>(workload_idx));

				Eigen::Map<Eigen::Vector3f> face_vertex0(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(0)));
				Eigen::Map<Eigen::Vector3f> face_vertex1(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(1)));
				Eigen::Map<Eigen::Vector3f> face_vertex2(vertex_position_indexer.GetDataPtr<float>(face_vertex_indices(2)));


#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Map<Eigen::Vector3f> face_vertices[] = {face_vertex0, face_vertex1, face_vertex2};
#endif
				// clip if all vertices are too near or too far
#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				bool have_vertex_within_clipping_range = false;
				for (auto& face_vertex: face_vertices) {
					have_vertex_within_clipping_range |= face_vertex.z() >= near_clipping_distance;
					have_vertex_within_clipping_range |= face_vertex.z() <= far_clipping_distance;
				}
				if (!have_vertex_within_clipping_range) {
					clipped_face_mask_ptr[workload_idx] = false;
					return;
				}
#else
				if ((face_vertex0.z() > far_clipping_distance &&
					 face_vertex1.z() > far_clipping_distance &&
					 face_vertex2.z() > far_clipping_distance) ||
					(face_vertex0.z() < near_clipping_distance &&
					 face_vertex1.z() < near_clipping_distance &&
					 face_vertex2.z() < near_clipping_distance)) {
					return;
				}
#endif

#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Vector2f normalized_face_vertices_xy[3];
				bool has_inliner_vertex = false;
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					perspective_transform.Project(face_vertices[i_vertex].x(), face_vertices[i_vertex].y(), face_vertices[i_vertex].z(),
					                              &normalized_face_vertices_xy[i_vertex].x(), &normalized_face_vertices_xy[i_vertex].y());
					has_inliner_vertex |= normalized_camera_space_xy_range.Contains(normalized_face_vertices_xy[i_vertex]);
				}
				if (!has_inliner_vertex) {
					clipped_face_mask_ptr[workload_idx] = false;
					return;
				}
#else
				Eigen::Vector2f normalized_face_vertex_xy0, normalized_face_vertex_xy1, normalized_face_vertex_xy2;

				perspective_transform.Project(face_vertex0.x(), face_vertex0.y(), face_vertex0.z(),
											  &normalized_face_vertex_xy0.x(), &normalized_face_vertex_xy0.y());
				perspective_transform.Project(face_vertex1.x(), face_vertex1.y(), face_vertex1.z(),
											  &normalized_face_vertex_xy1.x(), &normalized_face_vertex_xy1.y());
				perspective_transform.Project(face_vertex2.x(), face_vertex2.y(), face_vertex2.z(),
											  &normalized_face_vertex_xy2.x(), &normalized_face_vertex_xy2.y());

				if (!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy0) &&
					!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy1) &&
					!normalized_camera_space_xy_range.Contains(normalized_face_vertex_xy2)) {
					// face is outside of the view frustum's top/bottom/left/right boundaries
					return;
				}

#endif
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex0(normalized_face_vertex_indexer.GetDataPtr<float>(workload_idx));
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex1(normalized_face_vertex_indexer.GetDataPtr<float>(workload_idx) + 3);
				Eigen::Map<Eigen::Vector3f> normalized_face_vertex2(normalized_face_vertex_indexer.GetDataPtr<float>(workload_idx) + 6);
#ifdef NNRT_USE_SERIAL_TRIANGLE_VERTEX_HANDLING
				Eigen::Map<Eigen::Vector3f> normalized_face_vertices[] = {normalized_face_vertex0, normalized_face_vertex1, normalized_face_vertex2};
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					normalized_face_vertices[i_vertex].x() = normalized_face_vertices_xy[i_vertex].x();
					// important: flip the y-coordinate to reflect pixel space
					normalized_face_vertices[i_vertex].y() = normalized_face_vertices_xy[i_vertex].y();
					normalized_face_vertices[i_vertex].z() = face_vertices[i_vertex].z();
				}
#else
				normalized_face_vertex0.x() = normalized_face_vertex_xy0.x();
				normalized_face_vertex0.y() = normalized_face_vertex_xy0.y();
				normalized_face_vertex0.z() = face_vertex0.z();
				normalized_face_vertex1.x() = normalized_face_vertex_xy1.x();
				normalized_face_vertex1.y() = normalized_face_vertex_xy1.y();
				normalized_face_vertex1.z() = face_vertex1.z();
				normalized_face_vertex2.x() = normalized_face_vertex_xy2.x();
				normalized_face_vertex2.y() = normalized_face_vertex_xy2.y();
				normalized_face_vertex2.z() = face_vertex2.z();
#endif
			}
	);
}

inline void InitializeFragments(Fragments& fragments, const o3c::Device& device,
                                const int64_t& image_height, const int64_t& image_width,
                                const int faces_per_pixel) {
	// here we refer to intersection of the actual pixel's ray with a triangular face as "pixel"
	fragments.pixel_face_indices = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Int64, device);
	fragments.pixel_depths = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Float32, device);
	fragments.pixel_barycentric_coordinates = o3c::Tensor::Full({image_height, image_width, faces_per_pixel, 3}, -1, o3c::Float32, device);
	// float distance in the x/y camera plane in normalized camera-space coordinates, i.e. (-1,-1) to (1, 1), of each pixel-ray intersection
	// to faces_per_pixel of triangles closest to it along the z axis.
	fragments.pixel_face_distances = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Float32, device);
}

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshNaive(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> clipped_faces_mask,
		const open3d::core::SizeVector& image_size,
		float blur_radius, int faces_per_pixel, bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates,
		bool cull_back_faces
) {

	o3c::Device device = normalized_camera_space_face_vertices.GetDevice();

	o3tgk::TArrayIndexer<t_face_index> face_vertex_position_indexer(normalized_camera_space_face_vertices, 1);

	const auto face_count = static_cast<t_face_index>(normalized_camera_space_face_vertices.GetLength());

	const auto image_height = image_size[0];
	const auto image_width = image_size[1];
	const auto image_height_int = static_cast<int32_t>(image_height);
	const auto image_width_int = static_cast<int32_t>(image_width);
	const int64_t pixel_count = image_height * image_width;

	InitializeFragments(fragments, device, image_height, image_width, faces_per_pixel);

	if (normalized_camera_space_face_vertices.GetLength() == 0) {
		return;
	}

	// output data
	auto pixel_face_index_ptr = fragments.pixel_face_indices.template GetDataPtr<int64_t>();
	auto pixel_depth_ptr = fragments.pixel_depths.template GetDataPtr<float>();
	auto pixel_barycentric_coordinate_ptr = fragments.pixel_barycentric_coordinates.template GetDataPtr<float>();
	auto pixel_face_distance_ptr = fragments.pixel_face_distances.template GetDataPtr<float>();

	if (clipped_faces_mask.has_value()) {
		const bool* clipped_faces = clipped_faces_mask.value().get().template GetDataPtr<bool>();
		o3c::ParallelFor(
				device, pixel_count,
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					const auto v_image = static_cast<t_image_index>(workload_idx / image_width);
					const auto u_image = static_cast<t_image_index>(workload_idx % image_width);
					const float y_screen = ImageSpaceToNormalizedCameraSpace(v_image, image_height_int, image_width_int);
					const float x_screen = ImageSpaceToNormalizedCameraSpace(u_image, image_width_int, image_height_int);
					Eigen::Vector2f point_screen(x_screen, y_screen);
					RayFaceIntersection queue[MAX_POINTS_PER_PIXEL];
					int queue_size = 0;
					float queue_max_depth = -1000.f;
					int queue_max_depth_at = -1;

					// Loop through mesh faces.
					for (t_face_index i_face = 0; i_face < face_count; i_face++) {
						if (!clipped_faces[i_face]) {
							continue; // skip over clipped face
						}
						// Check if the point_screen ray goes through the face bounding box.
						// If it does, update the queue, queue_size, queue_max_depth and queue_max_depth_at in place.;
						UpdateQueueIfPixelInsideFace(
								face_vertex_position_indexer,
								i_face,
								queue,
								queue_size,
								queue_max_depth,
								queue_max_depth_at,
								blur_radius,
								point_screen,
								faces_per_pixel,
								perspective_correct_barycentric_coordinates,
								clip_barycentric_coordinates,
								cull_back_faces
						);

					}

#ifdef __CUDACC__
					core::functional::kernel::BubbleSort(queue, queue_size);
#else
					std::sort(std::begin(queue), std::begin(queue) + queue_size);
#endif

					int64_t fragment_index = workload_idx * faces_per_pixel;
					for (int i_pixel_face = 0; i_pixel_face < queue_size; i_pixel_face++) {
						pixel_face_index_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].face_index;
						pixel_depth_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].depth;
						pixel_face_distance_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].distance;
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 0] = queue[i_pixel_face].barycentric_coordinates.x();
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 1] = queue[i_pixel_face].barycentric_coordinates.y();
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 2] = queue[i_pixel_face].barycentric_coordinates.z();
					}
				}
		);
	} else {
		o3c::ParallelFor(
				device, pixel_count,
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					const auto v_image = static_cast<t_image_index>(workload_idx / image_width);
					const auto u_image = static_cast<t_image_index>(workload_idx % image_width);
					const float y_screen = ImageSpaceToNormalizedCameraSpace(v_image, image_height_int, image_width_int);
					const float x_screen = ImageSpaceToNormalizedCameraSpace(u_image, image_width_int, image_height_int);
					Eigen::Vector2f point_screen(x_screen, y_screen);
					RayFaceIntersection queue[MAX_POINTS_PER_PIXEL];
					int queue_size = 0;
					float queue_max_depth = -1000.f;
					int queue_max_depth_at = -1;

					// Loop through mesh faces.
					for (t_face_index i_face = 0; i_face < face_count; i_face++) {
						// Check if the point_screen ray goes through the face bounding box.
						// If it does, update the queue, queue_size, queue_max_depth and queue_max_depth_at in place.;
						UpdateQueueIfPixelInsideFace(
								face_vertex_position_indexer,
								i_face,
								queue,
								queue_size,
								queue_max_depth,
								queue_max_depth_at,
								blur_radius,
								point_screen,
								faces_per_pixel,
								perspective_correct_barycentric_coordinates,
								clip_barycentric_coordinates,
								cull_back_faces
						);

					}

#ifdef __CUDACC__
					core::functional::kernel::BubbleSort(queue, queue_size);
#else
					std::sort(std::begin(queue), std::begin(queue) + queue_size);
#endif
					int64_t fragment_index = workload_idx * faces_per_pixel;
					for (int i_pixel_face = 0; i_pixel_face < queue_size; i_pixel_face++) {
						pixel_face_index_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].face_index;
						pixel_depth_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].depth;
						pixel_face_distance_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].distance;
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 0] = queue[i_pixel_face].barycentric_coordinates.x();
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 1] = queue[i_pixel_face].barycentric_coordinates.y();
						pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 2] = queue[i_pixel_face].barycentric_coordinates.z();
					}
				}
		);
	}
}

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshFine(
		Fragments& fragments,
		const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::Tensor& bin_faces,
		const open3d::core::SizeVector& image_size,
		float blur_radius,
		int bin_side_length,
		int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates,
		bool clip_barycentric_coordinates,
		bool cull_back_faces
) {

	o3c::Device device = bin_faces.GetDevice();

	const int64_t image_height = image_size[0];
	const int64_t image_width = image_size[1];
	const auto image_height_int = static_cast<t_image_index>(image_height);
	const auto image_width_int = static_cast<t_image_index>(image_width);
	const int64_t pixel_count = image_height * image_width;

	const int bin_count_x = static_cast<int>(bin_faces.GetShape(1));
	const int bin_capacity = static_cast<int>(bin_faces.GetShape(2));

	const auto* bin_data = bin_faces.template GetDataPtr<t_face_index>();

	o3tgk::TArrayIndexer<t_face_index> face_vertex_position_indexer(normalized_camera_space_face_vertices, 1);

	InitializeFragments(fragments, device, image_height, image_width, faces_per_pixel);

	if (normalized_camera_space_face_vertices.GetLength() == 0) {
		return;
	}

	// output data
	auto pixel_face_index_ptr = fragments.pixel_face_indices.template GetDataPtr<int64_t>();
	auto pixel_depth_ptr = fragments.pixel_depths.template GetDataPtr<float>();
	auto pixel_barycentric_coordinate_ptr = fragments.pixel_barycentric_coordinates.template GetDataPtr<float>();
	auto pixel_face_distance_ptr = fragments.pixel_face_distances.template GetDataPtr<float>();

	o3c::ParallelFor(
			device,
			pixel_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const auto v_image = static_cast<t_image_index>(workload_idx / image_width);
				const auto u_image = static_cast<t_image_index>(workload_idx % image_width);

				const float y_screen = ImageSpaceToNormalizedCameraSpace(v_image, image_height_int, image_width_int);
				const float x_screen = ImageSpaceToNormalizedCameraSpace(u_image, image_width_int, image_height_int);

				const int v_bin = v_image / bin_side_length;
				const int u_bin = u_image / bin_side_length;

				Eigen::Vector2f point_screen(x_screen, y_screen);

				RayFaceIntersection queue[MAX_POINTS_PER_PIXEL];
				int queue_size = 0;
				float queue_max_depth = -1000.f;
				int queue_max_depth_at = -1;

				const t_face_index* current_bin_data = bin_data + (v_bin * bin_count_x + u_bin) * bin_capacity;

				// Loop through face indices in the pixel's bin.
				for (int i_bin_face_index = 0; i_bin_face_index < bin_capacity; i_bin_face_index++) {
					t_face_index face_index = current_bin_data[i_bin_face_index];
					if (face_index == -1) {
						// -1 is the sentinel value
#ifdef __CUDACC__
						continue;
#else
						break;
#endif
					}

					// Check if the point_screen ray goes through the face bounding box.
					// If it does, update the queue, queue_size, queue_max_depth and queue_max_depth_at in place.;
					UpdateQueueIfPixelInsideFace(
							face_vertex_position_indexer,
							i_bin_face_index,
							queue,
							queue_size,
							queue_max_depth,
							queue_max_depth_at,
							blur_radius,
							point_screen,
							faces_per_pixel,
							perspective_correct_barycentric_coordinates,
							clip_barycentric_coordinates,
							cull_back_faces
					);

				}
#ifdef __CUDACC__
				core::functional::kernel::BubbleSort(queue, queue_size);
#else
				std::sort(std::begin(queue), std::begin(queue) + queue_size);
#endif
				int64_t fragment_index = workload_idx * faces_per_pixel;
				for (int i_pixel_face = 0; i_pixel_face < queue_size; i_pixel_face++) {
					pixel_face_index_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].face_index;
					pixel_depth_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].depth;
					pixel_face_distance_ptr[fragment_index + i_pixel_face] = queue[i_pixel_face].distance;
					pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 0] = queue[i_pixel_face].barycentric_coordinates.x();
					pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 1] = queue[i_pixel_face].barycentric_coordinates.y();
					pixel_barycentric_coordinate_ptr[(fragment_index + i_pixel_face) * 3 + 2] = queue[i_pixel_face].barycentric_coordinates.z();
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void GridBin2dBoundingBoxes_Device(
		open3d::core::Tensor& bins,
		const open3d::core::Tensor& bounding_boxes,
		const open3d::core::Tensor& boxes_to_skip_mask,
		int image_height,
		int image_width,
		int bin_count_y,
		int bin_count_x,
		int bin_side_length,
		int bin_capacity,
		float half_pixel_x,
		float half_pixel_y
);

template<open3d::core::Device::DeviceType TDeviceType>
void GridBinFaces(
		open3d::core::Tensor& bin_faces, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> clipped_faces_mask,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int max_faces_per_bin
) {
	int image_height = static_cast<int>(image_size[0]);
	int image_width = static_cast<int>(image_size[1]);

	auto device = normalized_camera_space_face_vertices.GetDevice();
	int64_t face_count = normalized_camera_space_face_vertices.GetLength();
	o3tgk::NDArrayIndexer face_vertex_position_indexer(normalized_camera_space_face_vertices, 1);

	o3c::Tensor face_bounding_boxes({4, face_count}, o3c::Float32, device);
	o3c::Tensor face_skip_mask({face_count}, o3c::Bool, device);

	auto face_bounding_box_data = face_bounding_boxes.GetDataPtr<float>();
	auto face_skip_mask_data = face_skip_mask.GetDataPtr<bool>();

	// compute triangle bounding boxes
	if(clipped_faces_mask.has_value()) {
		const bool* clipped_faces = clipped_faces_mask.value().get().template GetDataPtr<bool>();
		o3c::ParallelFor(
				device, face_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					if(!clipped_faces[workload_idx]){
						face_skip_mask_data[workload_idx] = true;
						return;
					}
					auto face_vertices_data = face_vertex_position_indexer.GetDataPtr<float>(workload_idx);
					Eigen::Map<Eigen::Vector3f> face_vertex0(face_vertices_data);
					Eigen::Map<Eigen::Vector3f> face_vertex1(face_vertices_data + 3);
					Eigen::Map<Eigen::Vector3f> face_vertex2(face_vertices_data + 6);
					CalculateAndStoreFace2dBoundingBox(face_bounding_box_data, face_skip_mask_data, workload_idx, face_count, face_vertex0, face_vertex1,
					                                   face_vertex2, blur_radius);

				}
		);
	}else {
		o3c::ParallelFor(
				device, face_count,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
					auto face_vertices_data = face_vertex_position_indexer.GetDataPtr<float>(workload_idx);
					Eigen::Map<Eigen::Vector3f> face_vertex0(face_vertices_data);
					Eigen::Map<Eigen::Vector3f> face_vertex1(face_vertices_data + 3);
					Eigen::Map<Eigen::Vector3f> face_vertex2(face_vertices_data + 6);
					CalculateAndStoreFace2dBoundingBox(face_bounding_box_data, face_skip_mask_data, workload_idx, face_count, face_vertex0, face_vertex1,
					                                   face_vertex2, blur_radius);
				}
		);
	}


	const int bin_count_y = 1 + (image_height - 1) / bin_size;
	const int bin_count_x = 1 + (image_width - 1) / bin_size;

	bin_faces = o3c::Tensor::Full({bin_count_y, bin_count_x, max_faces_per_bin}, -1, o3c::Int32, device);


	const float half_normalized_camera_range_y = GetNormalizedCameraSpaceRange(image_width, image_height) / 2.f;
	const float half_normalized_camera_range_x = GetNormalizedCameraSpaceRange(image_height, image_width) / 2.f;

	const float half_pixel_y = half_normalized_camera_range_y / static_cast<float>(image_height);
	const float half_pixel_x = half_normalized_camera_range_x / static_cast<float>(image_width);

	GridBin2dBoundingBoxes_Device<TDeviceType>(bin_faces, face_bounding_boxes, face_skip_mask, image_height, image_width, bin_count_y, bin_count_x,
	                                           bin_size, max_faces_per_bin, half_pixel_x, half_pixel_y);


}

} // namespace nnrt::rendering::kernel