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
#include "rendering/kernel/RasterizeMesh.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/RayFaceIntersection.h"
#include "rendering/kernel/BubbleSort.h"
#include "core/PlatformIndependentAtomics.h"


namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;


namespace nnrt::rendering::kernel {


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

				//TODO: test whether serial version is slower
#define NNRT_USE_SERIAL_VERSION

#ifdef NNRT_USE_SERIAL_VERSION
				Eigen::Map<Eigen::Vector3f> face_vertices[] = {face_vertex0, face_vertex1, face_vertex2};
#endif
				// clip if all vertices are too near or too far
#ifdef NNRT_USE_SERIAL_VERSION
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

#ifdef NNRT_USE_SERIAL_VERSION
				Eigen::Vector2f normalized_face_vertices_xy[3];
				bool has_inliner_vertex = false;
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					perspective_transform.Project(face_vertices[i_vertex].x(), face_vertices[i_vertex].y(), face_vertices[i_vertex].z(),
					                              &normalized_face_vertices_xy[i_vertex].x(), &normalized_face_vertices_xy[i_vertex].y());
					has_inliner_vertex |= normalized_camera_space_xy_range.Contains(normalized_face_vertices_xy[i_vertex]);
				}
				if(!has_inliner_vertex){
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
#ifdef NNRT_USE_SERIAL_VERSION
				Eigen::Map<Eigen::Vector3f> normalized_face_vertices[] = {normalized_face_vertex0, normalized_face_vertex1, normalized_face_vertex2};
				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					normalized_face_vertices[i_vertex].x() = normalized_face_vertices_xy[i_vertex].x();
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
void RasterizeMeshNaive(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
) {

	o3c::Device device = normalized_camera_space_face_vertices.GetDevice();

	o3tgk::TArrayIndexer<t_face_index> face_vertex_position_indexer(normalized_camera_space_face_vertices, 1);

	const auto face_count = static_cast<t_face_index>(normalized_camera_space_face_vertices.GetLength());

	o3c::AssertTensorDtype(normalized_camera_space_face_vertices, o3c::Float32);

	const auto image_height = image_size[0];
	const auto image_width = image_size[1];
	const int64_t pixel_count = image_height * image_width;

	// here we refer to intersection of the actual pixel's ray with a triangular face as "pixel"
	fragments.pixel_face_indices = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Int64, device);
	fragments.pixel_depths = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Float32, device);
	fragments.pixel_barycentric_coordinates = o3c::Tensor::Full({image_height, image_width, faces_per_pixel, 3}, -1, o3c::Float32, device);
	// float distance in the x/y camera plane in normalized camera-space coordinates, i.e. (-1,-1) to (1, 1), of each pixel-ray intersection
	// to faces_per_pixel of triangles closest to it along the z axis.
	fragments.pixel_face_distances = o3c::Tensor::Full({image_height, image_width, faces_per_pixel}, -1, o3c::Float32, device);
	// indexers
	auto pixel_face_index_ptr = fragments.pixel_face_indices.template GetDataPtr<int64_t>();
	auto pixel_depth_ptr = fragments.pixel_depths.template GetDataPtr<float>();
	auto pixel_barycentric_coordinate_ptr = fragments.pixel_barycentric_coordinates.template GetDataPtr<float>();
	auto pixel_face_distance_ptr = fragments.pixel_face_distances.template GetDataPtr<float>();

	o3c::ParallelFor(
			device, pixel_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				const auto v_image = static_cast<t_image_index>(workload_idx / image_width);
				const auto u_image = static_cast<t_image_index>(workload_idx % image_width);
				const float y_screen = ImageToNormalizedCameraSpace(v_image, image_height, image_width);
				const float x_screen = ImageToNormalizedCameraSpace(u_image, image_width, image_height);
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
				BubbleSort(queue, queue_size);
#else
				std::sort(std::begin(queue), std::begin(queue) + queue_size);
#endif
				int fragment_index = workload_idx * faces_per_pixel;
				for (int i_pixel_face = 0; i_pixel_face < queue_size; i_pixel_face++){
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
void RasterizeMeshFine(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::Tensor& bin_faces,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
) {
	o3u::LogError("Not yet implemented!");
}

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshCoarse(
		open3d::core::Tensor& bin_faces, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int max_faces_per_bin) {
	o3u::LogError("Not yet implemented!");
}

} // namespace nnrt::rendering::kernel