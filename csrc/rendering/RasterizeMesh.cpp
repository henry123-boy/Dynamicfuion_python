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
// 3rd party
#include <open3d/utility/Logging.h>
#include <open3d/t/geometry/Utility.h>

// local
#include "rendering/RasterizeMesh.h"
#include "rendering/kernel/RasterizeMesh.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/CoordinateSystemConversions.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::rendering {


open3d::core::Tensor ExtractClippedFaceVerticesInNormalizedCameraSpace(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size, /* {height, width} */
		float near_clipping_distance /* = 0.0 */,
		float far_clipping_distance /* = INFINITY */
) {
	if (!camera_space_mesh.HasTriangleIndices() || !camera_space_mesh.HasVertexPositions()) {
		o3u::LogError("Argument mesh needs to have both triangle vertex indices and vertex positions. "
		              "Given mesh has triangle indices: {}, vertex positions: {}.",
		              camera_space_mesh.HasTriangleIndices() ? "true" : "false", camera_space_mesh.HasVertexPositions() ? "true" : "false");
	}
	if (near_clipping_distance < MIN_NEAR_CLIPPING_DISTANCE) {
		o3u::LogError("near_clipping_distance cannot be less than {}. Got {}.", MIN_NEAR_CLIPPING_DISTANCE, near_clipping_distance);
	}
	if (near_clipping_distance > far_clipping_distance) {
		o3u::LogError("near_clipping_distance cannot be greater than far_clipping_distance. Got {} and {}, respectively.",
		              near_clipping_distance, far_clipping_distance);
	}
	if (image_size.size() != 2) {
		o3u::LogError("image_size should be a SizeVector of size 2. Got size {}.", image_size.size());
	}

	const o3c::Tensor& vertex_positions_camera = camera_space_mesh.GetVertexPositions();
	const o3c::Tensor& triangle_vertex_indices = camera_space_mesh.GetTriangleIndices();
	o3c::AssertTensorDtype(vertex_positions_camera, o3c::Float32);
	o3c::AssertTensorDtype(triangle_vertex_indices, o3c::Int64);
	o3tg::CheckIntrinsicTensor(intrinsic_matrix);

	auto [normalized_intrinsic_matrix, normalized_xy_range] = kernel::IntrinsicsToNormalizedCameraSpaceAndRange(intrinsic_matrix, image_size);

	o3c::Tensor vertex_positions_clipped_normalized_camera;

	kernel::ExtractClippedFaceVerticesInNormalizedCameraSpace(vertex_positions_clipped_normalized_camera, vertex_positions_camera,
	                                                          triangle_vertex_indices, normalized_intrinsic_matrix, normalized_xy_range,
	                                                          near_clipping_distance, far_clipping_distance);

	return vertex_positions_clipped_normalized_camera;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeMesh(const open3d::core::Tensor& normalized_camera_space_face_vertices, const open3d::core::SizeVector& image_size,
              float blur_radius, int faces_per_pixel, int bin_size, int max_faces_per_bin, bool perspective_correct_barycentric_coordinates,
              bool clip_barycentric_coordinates, bool cull_back_faces) {


	if (faces_per_pixel > MAX_POINTS_PER_PIXEL) {
		o3u::LogError("Need faces_per_pixel <= {}. Got: {}", MAX_POINTS_PER_PIXEL, faces_per_pixel);
	}
	if (image_size.size() != 2) {
		o3u::LogError("image_size should be a SizeVector of size 2. Got size {}.", image_size.size());
	}

	kernel::Fragments fragments;
	if (bin_size > 0 && max_faces_per_bin > 0) {
		// Use coarse-to-fine rasterization
		o3c::Tensor bin_faces;
		kernel::RasterizeMeshCoarse(bin_faces,
		                            normalized_camera_space_face_vertices,
		                            image_size,
		                            blur_radius,
		                            bin_size,
									max_faces_per_bin);
		kernel::RasterizeMeshFine(fragments,
		                          normalized_camera_space_face_vertices,
		                          bin_faces,
		                          image_size,
		                          blur_radius,
		                          bin_size,
		                          faces_per_pixel,
		                          perspective_correct_barycentric_coordinates,
		                          clip_barycentric_coordinates,
		                          cull_back_faces);
	} else {

		// Use the naive per-pixel implementation
		kernel::RasterizeMeshNaive(fragments,
		                           normalized_camera_space_face_vertices,
		                           image_size,
		                           blur_radius,
		                           faces_per_pixel,
		                           perspective_correct_barycentric_coordinates,
		                           clip_barycentric_coordinates,
		                           cull_back_faces);

	}
	return fragments.ToTuple();
}



} // namespace nnrt::rendering