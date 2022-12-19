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
#include "rendering/kernel/ExtractClippedFaceVertices.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::rendering {

static void CheckMeshHasTrianglesAndVertices(const open3d::t::geometry::TriangleMesh& mesh) {
	if (!mesh.HasTriangleIndices() || !mesh.HasVertexPositions()) {
		utility::LogError("Argument mesh needs to have both triangle vertex indices and vertex positions. "
		              "Given mesh has triangle indices: {}, vertex positions: {}.",
		              mesh.HasTriangleIndices() ? "true" : "false", mesh.HasVertexPositions() ? "true" : "false");
	}

}

static void CheckClippingRangeAndImageSize(const open3d::core::SizeVector& image_size, /* {height, width} */
                                           float near_clipping_distance /* = 0.0 */,
                                           float far_clipping_distance /* = INFINITY */) {
	if (near_clipping_distance < MIN_NEAR_CLIPPING_DISTANCE) {
		utility::LogError("near_clipping_distance cannot be less than {}. Got {}.", MIN_NEAR_CLIPPING_DISTANCE, near_clipping_distance);
	}
	if (near_clipping_distance > far_clipping_distance) {
		utility::LogError("near_clipping_distance cannot be greater than far_clipping_distance. Got {} and {}, respectively.",
		                  near_clipping_distance, far_clipping_distance);
	}
	if (image_size.size() != 2) {
		utility::LogError("image_size should be a SizeVector of size 2. Got size {}.", image_size.size());
	}
}

open3d::core::Tensor GetMeshFaceVerticesNdc(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size, /* {height, width} */
		float near_clipping_distance /* = 0.0 */,
		float far_clipping_distance /* = INFINITY */
) {
	CheckMeshHasTrianglesAndVertices(camera_space_mesh);
	CheckClippingRangeAndImageSize(image_size, near_clipping_distance, far_clipping_distance);
	const o3c::Tensor& vertex_positions_camera = camera_space_mesh.GetVertexPositions();
	const o3c::Tensor& triangle_vertex_indices = camera_space_mesh.GetTriangleIndices();
	o3c::AssertTensorDtype(vertex_positions_camera, o3c::Float32);
	o3c::AssertTensorDtype(triangle_vertex_indices, o3c::Int64);
	o3tg::CheckIntrinsicTensor(intrinsic_matrix);

	auto [normalized_intrinsic_matrix, normalized_xy_range] = kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);

	o3c::Tensor vertex_positions_clipped_normalized_camera;

	kernel::MeshVerticesClippedToNdc(vertex_positions_clipped_normalized_camera, vertex_positions_camera,
	                                 triangle_vertex_indices, normalized_intrinsic_matrix, normalized_xy_range,
	                                 near_clipping_distance, far_clipping_distance);

	return vertex_positions_clipped_normalized_camera;
}


std::tuple<open3d::core::Tensor, open3d::core::Tensor> GetMeshNdcFaceVerticesAndClipMask(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size, /* {height, width} */
		float near_clipping_distance /* = 0.0 */,
		float far_clipping_distance /* = INFINITY */) {
	CheckMeshHasTrianglesAndVertices(camera_space_mesh);
	CheckClippingRangeAndImageSize(image_size, near_clipping_distance, far_clipping_distance);
	const o3c::Tensor& vertex_positions_camera = camera_space_mesh.GetVertexPositions();
	const o3c::Tensor& triangle_vertex_indices = camera_space_mesh.GetTriangleIndices();
	o3c::AssertTensorDtype(vertex_positions_camera, o3c::Float32);
	o3c::AssertTensorDtype(triangle_vertex_indices, o3c::Int64);
	o3tg::CheckIntrinsicTensor(intrinsic_matrix);

	auto [normalized_intrinsic_matrix, normalized_xy_range] = kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);
	o3c::Tensor vertex_positions_normalized_camera, clipped_face_mask;

	kernel::MeshDataAndClippingMaskToNdc(vertex_positions_normalized_camera, open3d::utility::nullopt, clipped_face_mask,
	                                     vertex_positions_camera, open3d::utility::nullopt, triangle_vertex_indices,
	                                     normalized_intrinsic_matrix, normalized_xy_range,
	                                     near_clipping_distance, far_clipping_distance);

	return std::make_tuple(vertex_positions_normalized_camera, clipped_face_mask);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> GetMeshFaceVerticesNdcAndNormalsNdcAndClipMask(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance /* = 0.0 */,
		float far_clipping_distance /* = INFINITY */) {
	CheckMeshHasTrianglesAndVertices(camera_space_mesh);
	if (!camera_space_mesh.HasVertexNormals()) {
		utility::LogError("Argument mesh needs to have vertex normals defined, but it does not. ");
	}
	CheckClippingRangeAndImageSize(image_size, near_clipping_distance, far_clipping_distance);
	const o3c::Tensor& vertex_positions_camera = camera_space_mesh.GetVertexPositions();
	const o3c::Tensor& vertex_normals_camera = camera_space_mesh.GetVertexNormals();
	const o3c::Tensor& triangle_vertex_indices = camera_space_mesh.GetTriangleIndices();
	o3c::AssertTensorDtype(vertex_positions_camera, o3c::Float32);
	o3c::AssertTensorDtype(triangle_vertex_indices, o3c::Int64);
	o3tg::CheckIntrinsicTensor(intrinsic_matrix);

	auto [normalized_intrinsic_matrix, normalized_xy_range] = kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);
	o3c::Tensor vertex_positions_normalized_camera, face_vertex_normals, clipped_face_mask;

	kernel::MeshDataAndClippingMaskToNdc(vertex_positions_normalized_camera, face_vertex_normals, clipped_face_mask,
	                                     vertex_positions_camera, vertex_normals_camera, triangle_vertex_indices,
	                                     normalized_intrinsic_matrix, normalized_xy_range,
	                                     near_clipping_distance, far_clipping_distance);

	return std::make_tuple(vertex_positions_normalized_camera, face_vertex_normals, clipped_face_mask);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeMesh(
		const open3d::core::Tensor& ndc_face_vertices,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> clipped_faces_mask,
		const open3d::core::SizeVector& image_size,
		float blur_radius_pixels,
		int faces_per_pixel,
		int bin_size/* = -1*/,
		int max_faces_per_bin/* = -1*/,
		bool perspective_correct_barycentric_coordinates /* = false */,
		bool clip_barycentric_coordinates /* = false */,
		bool cull_back_faces /* = true */
) {
	if (faces_per_pixel > MAX_POINTS_PER_PIXEL) {
		utility::LogError("Need faces_per_pixel <= {}. Got: {}", MAX_POINTS_PER_PIXEL, faces_per_pixel);
	}
	if (image_size.size() != 2) {
		utility::LogError("image_size should be a SizeVector of size 2. Got size {}.", image_size.size());
	}
	int64_t max_image_dimension = std::max(image_size[0], image_size[1]);

	if (bin_size == -1) {
		if (max_image_dimension <= 64) {
			bin_size = 8;
		} else {
			/*
			 * Heuristic based formula maps max_image_dimension to bin_size as follows:
			 * max_image_dimension < 64 -> 8
			 * 16 < max_image_dimension < 256 -> 16
			 * 256 < max_image_dimension < 512 -> 32
			 * 512 < max_image_dimension < 1024 -> 64
			 * 1024 < max_image_dimension < 2048 -> 128
			 */
			bin_size =
					static_cast<int>(std::pow(2, std::max(static_cast<int>(std::ceil(std::log2(static_cast<double>(max_image_dimension)))) - 4, 4)));
		}
	}

	if (bin_size != 0) {
		int bins_along_max_dimension = 1 + (max_image_dimension - 1) / bin_size;
		if (bins_along_max_dimension >= MAX_BINS_ALONG_IMAGE_DIMENSION) {
			utility::LogError("The provided bin_size is too small: computed bin count along maximum dimension is {}, this has to be < {}.",
			                  bins_along_max_dimension, MAX_BINS_ALONG_IMAGE_DIMENSION);
		}
	}

	if (max_faces_per_bin == -1) {
		if (clipped_faces_mask.has_value()) {
			int32_t unclipped_face_count = static_cast<int32_t>(clipped_faces_mask.value().get().NonZero().GetShape(1));
			max_faces_per_bin = std::max(10000, unclipped_face_count / 5);
		} else {
			max_faces_per_bin = std::max(10000, static_cast<int>(ndc_face_vertices.GetLength()) / 5);
		}
	}

	o3c::AssertTensorDtype(ndc_face_vertices, o3c::Float32);


	kernel::Fragments fragments;
    float blur_radius_ndc = kernel::ImageSpaceDistanceToNdc(blur_radius_pixels, image_size[0], image_size[1]);

	if (bin_size > 0 && max_faces_per_bin > 0) {
		// Use coarse-to-fine rasterization
		o3c::Tensor bin_faces;
		kernel::GridBinFaces(bin_faces,
                             ndc_face_vertices,
                             clipped_faces_mask,
                             image_size,
                             blur_radius_ndc,
                             bin_size,
                             max_faces_per_bin);

		kernel::RasterizeMeshFine(fragments,
                                  ndc_face_vertices,
                                  bin_faces,
                                  image_size,
                                  blur_radius_ndc,
                                  bin_size,
                                  faces_per_pixel,
                                  perspective_correct_barycentric_coordinates,
                                  clip_barycentric_coordinates,
                                  cull_back_faces);
	} else {

		// Use the naive per-pixel implementation
		kernel::RasterizeMeshNaive(fragments,
                                   ndc_face_vertices,
                                   clipped_faces_mask,
                                   image_size,
                                   blur_radius_ndc,
                                   faces_per_pixel,
                                   perspective_correct_barycentric_coordinates,
                                   clip_barycentric_coordinates,
                                   cull_back_faces);

	}
	return fragments.ToTuple();
}


} // namespace nnrt::rendering