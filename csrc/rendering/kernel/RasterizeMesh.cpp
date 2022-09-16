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
#include "rendering/kernel/RasterizeMesh.h"
#include "core/DeviceSelection.h"
#include "CoordinateSystemConversions.h"

namespace o3c = open3d::core;

namespace nnrt::rendering::kernel {


void ExtractClippedFaceVerticesInNormalizedCameraSpace(open3d::core::Tensor& vertex_positions_clipped_normalized_camera,
                                                       const open3d::core::Tensor& vertex_positions_camera,
                                                       const open3d::core::Tensor& triangle_vertex_indices,
                                                       const open3d::core::Tensor& normalized_intrinsic_matrix,
                                                       kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
                                                       float near_clipping_distance,
                                                       float far_clipping_distance) {
	core::ExecuteOnDevice(
			vertex_positions_camera.GetDevice(),
			[&] {
				ExtractClippedFaceVerticesInNormalizedCameraSpace<o3c::Device::DeviceType::CPU>(
						vertex_positions_clipped_normalized_camera, vertex_positions_camera, triangle_vertex_indices, normalized_intrinsic_matrix,
						normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ExtractClippedFaceVerticesInNormalizedCameraSpace<o3c::Device::DeviceType::CUDA>(
								vertex_positions_clipped_normalized_camera, vertex_positions_camera, triangle_vertex_indices,
								normalized_intrinsic_matrix, normalized_camera_space_xy_range, near_clipping_distance, far_clipping_distance
						);
				);
			}
	);
}

void RasterizeMeshNaive(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::SizeVector& image_size,
		float blur_radius, int faces_per_pixel, bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates,
		bool cull_back_faces
) {
	o3c::Device device = normalized_camera_space_face_vertices.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				RasterizeMeshNaive<o3c::Device::DeviceType::CPU>(
						fragments, normalized_camera_space_face_vertices, image_size, blur_radius, faces_per_pixel,
						perspective_correct_barycentric_coordinates, clip_barycentric_coordinates, cull_back_faces
				);
			},
			[&] {
				NNRT_IF_CUDA(
						RasterizeMeshNaive<o3c::Device::DeviceType::CUDA>(
								fragments, normalized_camera_space_face_vertices, image_size, blur_radius, faces_per_pixel,
								perspective_correct_barycentric_coordinates, clip_barycentric_coordinates, cull_back_faces
						);
				);
			}
	);
}

void
RasterizeMeshFine(
		Fragments& fragments, const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::Tensor& bin_faces,
		const open3d::core::SizeVector& image_size, float blur_radius, int bin_size, int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
) {
	o3c::Device device = normalized_camera_space_face_vertices.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				RasterizeMeshFine<o3c::Device::DeviceType::CPU>(
						fragments, normalized_camera_space_face_vertices, bin_faces, image_size, blur_radius, bin_size, faces_per_pixel,
						perspective_correct_barycentric_coordinates, clip_barycentric_coordinates, cull_back_faces
				);
			},
			[&] {
				NNRT_IF_CUDA(
						RasterizeMeshFine<o3c::Device::DeviceType::CUDA>(
								fragments, normalized_camera_space_face_vertices, bin_faces, image_size, blur_radius, bin_size, faces_per_pixel,
								perspective_correct_barycentric_coordinates, clip_barycentric_coordinates, cull_back_faces
						);
				);
			}
	);
}

void GridBinFaces(
		open3d::core::Tensor& bin_faces, const open3d::core::Tensor& normalized_camera_space_face_vertices,
		const open3d::core::SizeVector& image_size, const float blur_radius, const int bin_size, const int max_faces_per_bin
) {
	o3c::Device device = normalized_camera_space_face_vertices.GetDevice();
	core::ExecuteOnDevice(
			device,
			[&] {
				GridBinFaces<o3c::Device::DeviceType::CPU>
						(bin_faces, normalized_camera_space_face_vertices, image_size, blur_radius, bin_size, max_faces_per_bin);
			},
			[&] {
				NNRT_IF_CUDA(GridBinFaces<o3c::Device::DeviceType::CUDA>
						             (bin_faces, normalized_camera_space_face_vertices, image_size, blur_radius, bin_size, max_faces_per_bin););
			}
	);
}


} // namespace nnrt::rendering::kernel