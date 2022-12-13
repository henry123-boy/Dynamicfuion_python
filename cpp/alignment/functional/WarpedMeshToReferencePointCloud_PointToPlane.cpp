//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/core/Dispatch.h>

// local includes
#include "core/functional/Masking.h"
#include "geometry/GraphWarpField.h"
#include "rendering/functional/InterpolateFaceAttributes.h"
#include "rendering/RasterizeMesh.h"
#include "WarpedMeshToReferencePointCloud_PointToPlane.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment::functional {
std::tuple<open3d::core::Tensor, PartialDifferentiationState>
ComputeReferencePointToRenderedPlaneDistances(const nnrt::geometry::GraphWarpField& warp_field,
                                              const open3d::t::geometry::TriangleMesh& canonical_mesh,
                                              const open3d::t::geometry::Image& reference_color_image,
                                              const open3d::t::geometry::PointCloud& reference_point_cloud,
                                              const open3d::core::Tensor& intrinsics,
                                              const open3d::core::Tensor& extrinsics,
                                              const open3d::core::Tensor& anchors,
                                              const open3d::core::Tensor& weights) {
	PartialDifferentiationState differentiation_state;
	o3c::SizeVector image_size = {reference_color_image.AsTensor().GetShape(0), reference_color_image.AsTensor().GetShape(1)};
	o3tg::TriangleMesh warped_mesh = warp_field.WarpMesh(canonical_mesh, anchors, weights, true, extrinsics);
	auto [extracted_face_vertices, clipped_face_mask] =
			rendering::MeshFaceVerticesAndClipMaskToNdc(warped_mesh, intrinsics, image_size);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			rendering::RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0, 1, -1, -1, true, false, true);
	auto vertex_normals = warped_mesh.GetVertexNormals();
	auto triangle_indices = warped_mesh.GetTriangleIndices();
	auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));
	auto rendered_normals = rendering::functional::InterpolateFaceAttributes(pixel_face_indices, pixel_barycentric_coordinates, face_vertex_normals);
	differentiation_state.rendered_normals = rendered_normals;
	int64_t pixel_attribute_count = rendered_normals.GetShape(3);
	// both statements below assume 1x faces per pixel, neet to get first slice along pixel-face axis otherwise
	rendered_normals = rendered_normals.Reshape({image_size[0] * image_size[1], pixel_attribute_count});
	pixel_depths = pixel_depths.Reshape({image_size[0], image_size[1]});


	auto negative_ones = -1 * o3c::Tensor::Ones(pixel_depths.GetShape(), pixel_depths.GetDtype(), pixel_depths.GetDevice());
	o3c::Tensor rendered_point_mask = pixel_depths.IsClose(negative_ones, 0, 0).LogicalNot();
	core::functional::SetMaskedToValue(pixel_depths, rendered_point_mask, 0.f);


	o3tg::PointCloud rendered_point_cloud =
			o3tg::PointCloud::CreateFromDepthImage(o3tg::Image(pixel_depths), intrinsics, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")),
			                                       1.0, 1000.0f);

	o3c::Tensor rendered_points = rendered_point_cloud.GetPointPositions().GetItem(o3c::TensorKey::IndexTensor(rendered_point_mask));
	const o3c::Tensor& reference_points = reference_point_cloud.GetPointPositions();
	o3c::Tensor reference_to_rendered_point_vectors = rendered_points - reference_points;
	differentiation_state.reference_to_rendered_vectors = reference_to_rendered_point_vectors;
	// all masked values should produce distances of 0.
	o3c::Tensor distances = (rendered_normals * reference_to_rendered_point_vectors).Sum({1});
	return std::make_tuple(distances,differentiation_state);
}

} // namespace nnrt::alignment::functional

