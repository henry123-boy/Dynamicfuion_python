//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/19/22.
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
// stdlib includes

// third-party includes
#include <open3d/t/geometry/Utility.h>

// local includes
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/functional/kernel/ExtractClippedFaceVertices.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::rendering::functional{
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

	auto [ndc_intrinsic_matrix, ndc_xy_range] =
            rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);
	o3c::Tensor vertex_positions_normalized_camera, clipped_face_mask;

    kernel::MeshDataAndClippingMaskToNdc(vertex_positions_normalized_camera,
                                         open3d::utility::nullopt,
                                         clipped_face_mask,
                                         open3d::utility::nullopt,
                                         {vertex_positions_camera},
                                         open3d::utility::nullopt,
                                         {triangle_vertex_indices},
                                         ndc_intrinsic_matrix, ndc_xy_range,
                                         near_clipping_distance, far_clipping_distance);

	return std::make_tuple(vertex_positions_normalized_camera, clipped_face_mask);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> GetMeshNdcFaceVerticesAndClipMask(
        const std::vector<open3d::t::geometry::TriangleMesh>& camera_space_meshes,
        const open3d::core::Tensor& intrinsic_matrix,
        const open3d::core::SizeVector& image_size,
        float near_clipping_distance,
        float far_clipping_distance
) {
    CheckClippingRangeAndImageSize(image_size, near_clipping_distance, far_clipping_distance);
    std::vector<o3c::Tensor> vertex_position_sets;
    std::vector<o3c::Tensor> triangle_vertex_index_sets;

    for(const auto& mesh : camera_space_meshes){
        CheckMeshHasTrianglesAndVertices(mesh);
        const o3c::Tensor& vertex_positions_camera = mesh.GetVertexPositions();
	    const o3c::Tensor& triangle_vertex_indices = mesh.GetTriangleIndices();
	    o3c::AssertTensorDtype(vertex_positions_camera, o3c::Float32);
	    o3c::AssertTensorDtype(triangle_vertex_indices, o3c::Int64);
        vertex_position_sets.push_back(vertex_positions_camera);
        triangle_vertex_index_sets.push_back(triangle_vertex_indices);
    }

    auto [ndc_intrinsic_matrix, ndc_xy_range] =
            rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);
	o3c::Tensor vertex_positions_normalized_camera, clipped_face_mask, face_counts;

    kernel::MeshDataAndClippingMaskToNdc(vertex_positions_normalized_camera,
                                         open3d::utility::nullopt,
                                         clipped_face_mask,
                                         face_counts,
                                         vertex_position_sets,
                                         open3d::utility::nullopt,
                                         triangle_vertex_index_sets,
                                         ndc_intrinsic_matrix,
                                         ndc_xy_range,
                                         near_clipping_distance,
                                         far_clipping_distance);


    return std::make_tuple(vertex_positions_normalized_camera, clipped_face_mask, face_counts);
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

	auto [ndc_intrinsic_matrix, ndc_xy_range] =
            rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);
	o3c::Tensor vertex_positions_normalized_camera, face_vertex_normals, clipped_face_mask;
    const std::vector<open3d::core::Tensor> vnc_vector = {vertex_normals_camera};

    kernel::MeshDataAndClippingMaskToNdc(vertex_positions_normalized_camera,
                                         face_vertex_normals,
                                         clipped_face_mask,
                                         open3d::utility::nullopt,
                                         {vertex_positions_camera},
                                         vnc_vector,
                                         {triangle_vertex_indices},
                                         ndc_intrinsic_matrix,
                                         ndc_xy_range,
                                         near_clipping_distance,
                                         far_clipping_distance);

	return std::make_tuple(vertex_positions_normalized_camera, face_vertex_normals, clipped_face_mask);
}



} // namespace nnrt::rendering::functional