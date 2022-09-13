//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/5/22.
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

#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include "CoordinateSystemConversions.h"


namespace nnrt::rendering::kernel {

struct Fragments {
	open3d::core::Tensor pixel_faces, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances;

	std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> ToTuple() {
		return std::make_tuple(pixel_faces, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances);
	}
};

using t_image_index = int32_t;

void ExtractClippedFaceVerticesInNormalizedCameraSpace(open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
                                                       const open3d::core::Tensor& vertex_positions_camera,
                                                       const open3d::core::Tensor& triangle_vertex_indices,
                                                       const open3d::core::Tensor& normalized_camera_space_matrix,
                                                       nnrt::rendering::kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
                                                       float near_clipping_distance,
                                                       float far_clipping_distance);

template<open3d::core::Device::DeviceType TDeviceType>
void ExtractClippedFaceVerticesInNormalizedCameraSpace(open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
                                                       const open3d::core::Tensor& vertex_positions_camera,
                                                       const open3d::core::Tensor& triangle_vertex_indices,
                                                       const open3d::core::Tensor& normalized_camera_space_matrix,
                                                       AxisAligned2dBoundingBox normalized_camera_space_xy_range,
                                                       float near_clipping_distance,
                                                       float far_clipping_distance);

void RasterizeMeshNaive(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
);

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshNaive(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
);


void RasterizeMeshFine(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::Tensor& bin_faces,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
);

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshFine(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::Tensor& bin_faces,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
);

void RasterizeMeshCoarse(
		open3d::core::Tensor& bin_faces, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int max_faces_per_bin
);

template<open3d::core::Device::DeviceType TDeviceType>
void RasterizeMeshCoarse(
		open3d::core::Tensor& bin_faces, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int max_faces_per_bin
);


} // namespace nnrt::rendering::kernel