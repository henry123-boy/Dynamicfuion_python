//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/16/22.
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

// open3d
#include <open3d/t/geometry/PointCloud.h>

// local
#include "rendering/DeformableMeshRenderToRgbdImageFitter.h"
#include "rendering/RasterizeMesh.h"
#include "rendering/functional/InterpolateFaceAttributes.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

namespace nnrt::rendering {

void DeformableMeshRenderToRgbdImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field, const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::PointCloud& reference_point_cloud,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix
) const {
	auto [anchors, weights] = warp_field.PrecomputeAnchorsAndWeights(canonical_mesh, nnrt::geometry::AnchorComputationMethod::SHORTEST_PATH);

}

void DeformableMeshRenderToRgbdImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::Image& reference_depth_image,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix, float depth_scale, float depth_max
) const {
	open3d::t::geometry::PointCloud point_cloud =
			o3tg::PointCloud::CreateFromDepthImage(reference_depth_image, intrinsic_matrix,
			                                       o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), depth_scale, depth_max);
	FitToImage(warp_field, canonical_mesh, reference_color_image, point_cloud, intrinsic_matrix, extrinsic_matrix);
}

void
DeformableMeshRenderToRgbdImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field, const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::RGBDImage& reference_image,
		const open3d::core::Tensor& intrinsic_matrix, const open3d::core::Tensor& extrinsic_matrix,
		float depth_scale, float depth_max
) const {
	FitToImage(warp_field, canonical_mesh, reference_image.color_, reference_image.depth_, intrinsic_matrix, extrinsic_matrix,
	           depth_scale, depth_max);
}

open3d::core::Tensor DeformableMeshRenderToRgbdImageFitter::ComputeResiduals(nnrt::geometry::GraphWarpField& warp_field,
                                                                             const open3d::t::geometry::TriangleMesh& canonical_mesh,
                                                                             const open3d::t::geometry::Image& reference_color_image,
                                                                             const open3d::t::geometry::PointCloud& reference_point_cloud,
                                                                             const open3d::core::Tensor& intrinsics,
                                                                             const open3d::core::Tensor& extrinsics,
                                                                             const open3d::core::Tensor& anchors,
                                                                             const open3d::core::Tensor& weights) const {
	o3c::SizeVector image_size = {reference_color_image.AsTensor().GetShape(0), reference_color_image.AsTensor().GetShape(1)};
	o3tg::TriangleMesh warped_mesh = warp_field.WarpMesh(canonical_mesh, anchors, weights, true, extrinsics);
	auto [extracted_face_vertices, clipped_face_mask] =
			MeshFaceVerticesAndClipMaskToRaySpace(warped_mesh, intrinsics, image_size);
	auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
			RasterizeMesh(extracted_face_vertices, clipped_face_mask, image_size, 0, 1, -1, -1, true, false, true);
	auto vertex_normals = warped_mesh.GetVertexNormals();
	auto triangle_indices = warped_mesh.GetTriangleIndices();
	auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));
	auto rendered_normals = functional::InterpolateFaceAttributes(pixel_face_indices, pixel_barycentric_coordinates, face_vertex_normals);
	int64_t pixel_attribute_count = rendered_normals.GetShape(3);
	// both statements below assume 1 x faces per pixel, neet to get first slice along pixel-face axis otherwise
	rendered_normals = rendered_normals.Reshape({image_size[0] * image_size[1], pixel_attribute_count});
	pixel_depths = pixel_depths.Reshape({image_size[0], image_size[1]});
	o3tg::PointCloud rendered_point_cloud =
			o3tg::PointCloud::CreateFromDepthImage(o3tg::Image(pixel_depths), intrinsics, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")),
			                                       1.0, 1000.0f);


}


} // nnrt::rendering