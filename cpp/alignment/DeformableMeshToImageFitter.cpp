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
#include <open3d/core/Dispatch.h>
#include <open3d/t/io/ImageIO.h>

// local
#include "core/functional/Masking.h"
#include "core/linalg/SolveCholesky.h"
#include "alignment/DeformableMeshToImageFitter.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/PointToPlaneDistances.h"
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "alignment/functional/WarpedVertexAndNormalJacobians.h"
#include "alignment/functional/RasterizedVertexAndNormalJacobians.h"
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/linalg/Rodrigues.h"

namespace o3tio = open3d::t::io;

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::alignment {

DeformableMeshToImageFitter::DeformableMeshToImageFitter(
		int maximal_iteration_count /* = 100*/,
		float minimal_update_threshold /* = 1e-6*/,
		bool use_perspective_correction /* = false*/,
		float max_depth /* = 10.f*/,
		bool use_tukey_penalty /* = false*/,
		float tukey_penalty_cutoff_cm /* = 0.01*/
) : max_iteration_count(maximal_iteration_count),
    min_update_threshold(minimal_update_threshold),
    max_depth(max_depth),
    use_perspective_correction(use_perspective_correction),
    use_tukey_penalty(use_tukey_penalty),
    tukey_penalty_cutoff_cm(tukey_penalty_cutoff_cm) {}

void DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::PointCloud& reference_point_cloud,
		const open3d::core::Tensor& reference_point_mask,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix,
		const open3d::core::SizeVector& rendering_image_size
) const {
	//TODO: make parameter EUCLIDEAN/SHORTEST_PATH, optionally set in constructor
	auto [warp_anchors, warp_weights] = warp_field.PrecomputeAnchorsAndWeights(canonical_mesh,
	                                                                           nnrt::geometry::AnchorComputationMethod::EUCLIDEAN);

	int iteration = 0;
	float maximum_update = std::numeric_limits<float>::max();

	auto [ndc_intrinsic_matrix, ndc_xy_range] =
			rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, rendering_image_size);


	while (iteration < max_iteration_count && maximum_update > min_update_threshold) {
		o3tg::TriangleMesh
				warped_mesh = warp_field.WarpMesh(canonical_mesh, warp_anchors, warp_weights, true, extrinsic_matrix);

		auto [extracted_face_vertices, clipped_face_mask] =
				nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(
						warped_mesh, intrinsic_matrix, rendering_image_size, 0.0, 10.0
				);

		//TODO: add an obvious optional optimization in the case perspective-correct barycentrics are used: output the
		// distorted barycentric coordinates and reuse them, instead of recomputing them, in RasterizedVertexAndNormalJacobians
		std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
				nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, rendering_image_size, 0.5f, 1, -1, -1,
				                                       this->use_perspective_correction, false, true);;
		auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

		//__DEBUG
		// bool draw_depth = true;
		// if (draw_depth) {
		// 	auto pd_tmp = pixel_depths.Clone();
		// 	nnrt::core::functional::ReplaceValue(pd_tmp, -1.0f, 10.0f);
		// 	float minimum_depth = pd_tmp.Min({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
		// 	float maximum_depth = pixel_depths.Max({0, 1}).To(o3c::Device("CPU:0")).ToFlatVector<float>()[0];
		// 	nnrt::core::functional::ReplaceValue(pd_tmp, 10.0f, minimum_depth);
		// 	pd_tmp = 255.f - ((pd_tmp - minimum_depth) * 255.f / (maximum_depth - minimum_depth));
		// 	o3tg::Image stretched_depth_image(pd_tmp.To(o3c::UInt8));
		// 	o3tio::WriteImage("/home/algomorph/Builds/NeuralTracking/cmake-build-debug/cpp/tests/test_data/images/__debug_depth_1-node.png", stretched_depth_image);
		// }

		//__DEBUG
		int debug_start_row = 46;
		int debug_start_col = 45;
		int debug_end_row = 54;
		int debug_end_col = 55;

		// __DEBUG
		// auto center_faces = pixel_face_indices.Slice(0, debug_start_row, debug_end_row).Slice(1, debug_start_col, debug_end_col).Clone();


		// compute residuals r, retain rasterized points & global mask relevant for energy function being minimized
		o3tg::PointCloud rasterized_point_cloud;
		o3c::Tensor residual_mask;
		// [PX]; PX = W * H
		o3c::Tensor residuals =
				this->ComputeResiduals(
						rasterized_point_cloud, residual_mask, warped_mesh, pixel_face_indices,
						pixel_barycentric_coordinates, pixel_depths,
						reference_color_image, reference_point_cloud, reference_point_mask,
						intrinsic_matrix
				);

		//TODO: revise termination conditions to check the residual magnitudes / energy somehow

		//__DEBUG
		auto center_rasterized_point_positions = rasterized_point_cloud.GetPointPositions().Reshape({100, 100, 3})
		                                                               .Slice(0, debug_start_row, debug_end_row)
		                                                               .Slice(1, debug_start_col, debug_end_col).Clone();
		auto center_rasterized_point_normals = rasterized_point_cloud.GetPointNormals().Reshape({100, 100, 3})
		                                                             .Slice(0, debug_start_row, debug_end_row)
		                                                             .Slice(1, debug_start_col, debug_end_col).Clone();
		auto center_reference_point_positions = reference_point_cloud.GetPointPositions().Reshape({100, 100, 3})
		                                                             .Slice(0, debug_start_row, debug_end_row)
		                                                             .Slice(1, debug_start_col, debug_end_col).Clone();
		auto center_residuals = residuals.Reshape({100, 100}).Slice(0, debug_start_row, debug_end_row)
		                                 .Slice(1, debug_start_col, debug_end_col).Clone();
		// auto center_masks = residual_mask.Reshape({100,100})
		// .Slice(0, debug_start_row, debug_end_row).Slice(1, debug_start_col, debug_end_col).Clone();

		// compute warped vertex and normal jacobians wrt. delta rotations and jacobians
		// [V X A/V X 4], [V X A/V X 3]
		auto [warped_vertex_position_jacobians, warped_vertex_normal_jacobians] =
				functional::WarpedVertexAndNormalJacobians(canonical_mesh, warp_field, warp_anchors, warp_weights);


		// compute rasterized vertex and normal jacobians
		// [W x H x 3 x 9], [W x H x 30]; 30 = 3x9, 1x3
		//TODO: probably better to explicitly separate the rasterized_vertex_normal_jacobians ([W x H x 3x9]) from the barycentrics ([W x H x 1x3])
		// for clarity's sake. Also, it makes sense to be explicit that it's not the full "rasterized_vertex_normal_jacobians"
		// (missing the barycentrics part), or at least insert some comments about this
		auto [rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians] =
				functional::RasterizedVertexAndNormalJacobians(
						warped_mesh, pixel_face_indices,
						pixel_barycentric_coordinates,
						ndc_intrinsic_matrix, this->use_perspective_correction
				);

		//__DEBUG
		auto center_rvp_jacobians = rasterized_vertex_position_jacobians.Slice(0, debug_start_row, debug_end_row)
		                                                                .Slice(1, debug_start_col, debug_end_col).Clone();


		// compute J, i.e. sparse Jacobian at every pixel w.r.t. every node delta
		o3c::Tensor point_map_vectors = rasterized_point_cloud.GetPointPositions() - reference_point_cloud.GetPointPositions();
		o3c::Tensor rasterized_normals = rasterized_point_cloud.GetPointNormals();
		//__DEBUG
		// auto center_point_map_vectors = point_map_vectors.Reshape({100, 100, 3}).Slice(0, debug_start_row, debug_end_row).Slice(1, debug_start_col, debug_end_col).Clone();

		o3c::Tensor pixel_jacobians, pixel_node_jacobian_counts, node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts;

		//__DEBUG
		kernel::ComputePixelVertexAnchorJacobiansAndNodeAssociations(
				pixel_jacobians, pixel_node_jacobian_counts, node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts,
				rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
				warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
				point_map_vectors, rasterized_normals, residual_mask, pixel_face_indices,
				warped_mesh.GetTriangleIndices(), warp_anchors, warp_field.nodes.GetLength(), false, 0.01
		);

		//__DEBUG
		// auto anchor_count = warp_anchors.GetShape(1);
		// ==== w/o dropping anchor data beyond first node-anchor (multiple-node-case) ====
		// auto center_pixel_jacobians = pixel_jacobians.Reshape({100,100, anchor_count*3, 6}).Slice(0, debug_start_row, debug_end_row).Slice(1, debug_start_col, debug_end_col).Clone();
		// auto center_pixel_rotation_jacobians = center_pixel_jacobians.Slice(3,0,3).Clone();
		// auto center_pixel_translation_jacobians = center_pixel_jacobians.Slice(3,3,6).Clone();
		// auto center_pixel_rotation_gradients = center_pixel_rotation_jacobians.Sum({2,3});
		// auto center_pixel_translation_gradients = center_pixel_translation_jacobians.Sum({2,3});
		// ==== w/ dropping anchor data beyond first node-anchor (single-node-case) ====
		auto center_pixel_jacobians = pixel_jacobians.Slice(1, 0, 1).Reshape({100, 100, 6})
		                                             .Slice(0, debug_start_row, debug_end_row).Slice(1, debug_start_col, debug_end_col).Clone();
		auto center_pixel_rotation_jacobians = center_pixel_jacobians.Slice(2, 0, 3).Clone();
		auto center_pixel_translation_jacobians = center_pixel_jacobians.Slice(2, 3, 6).Clone();
		auto center_pixel_rotation_gradients = center_pixel_rotation_jacobians.Sum({2});
		auto center_pixel_translation_gradients = center_pixel_translation_jacobians.Sum({2});



		// compute (J^T)J, i.e. hessian approximation, in block-diagonal form
		open3d::core::Tensor hessian_approximation_blocks;
		kernel::ComputeHessianApproximationBlocks_UnorderedNodePixels(
				hessian_approximation_blocks, pixel_jacobians,
				node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts
		);

		// compute -(J^T)r
		o3c::Tensor negative_gradient;
		int max_anchor_count_per_vertex = warp_anchors.GetShape(1);
		kernel::ComputeNegativeGradient_UnorderedNodePixels(
				negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices_jagged,
				node_pixel_jacobian_counts, max_anchor_count_per_vertex
		);

		open3d::core::Tensor motion_updates;
		core::linalg::SolveCholeskyBlockDiagonal(motion_updates, hessian_approximation_blocks, negative_gradient);

		motion_updates = motion_updates.Reshape({motion_updates.GetShape(0) / 6, 6});

		// convert rotation axis-angle vectors to matrices
		auto rotation_matrix_updates = core::linalg::AxisAngleVectorsToMatricesRodrigues(motion_updates.Slice(1, 0, 3));

		// apply motion updates
		warp_field.TranslateNodes(motion_updates.Slice(1, 3, 6));
		warp_field.RotateNodes(rotation_matrix_updates);


		iteration++;
	}
}

void
DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::Image& reference_depth_image,
		const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix,
		float depth_scale
) const {
	o3c::Tensor identity_extrinsics = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));


	o3c::Tensor points, reference_point_depth_mask;
	nnrt::geometry::functional::UnprojectDepthImageWithoutFiltering(
			points, reference_point_depth_mask, reference_depth_image.AsTensor(), intrinsic_matrix, identity_extrinsics,
			depth_scale, this->max_depth, false
	);


	open3d::t::geometry::PointCloud point_cloud(points);

	o3c::Tensor final_reference_point_mask;
	if (reference_image_mask.has_value()) {
		final_reference_point_mask = reference_point_depth_mask.LogicalAnd(
				reference_image_mask.value().get().Reshape(reference_point_depth_mask.GetShape())
		);
	} else {
		final_reference_point_mask = reference_point_depth_mask;
	}

	o3c::SizeVector rendering_image_size{reference_depth_image.GetRows(), reference_depth_image.GetCols()};
	//TODO: employ the min_update_threshold termination condition as well
	for (int i_iteration = 0; i_iteration < this->max_iteration_count; i_iteration++) {
		FitToImage(warp_field, canonical_mesh, reference_color_image, point_cloud, final_reference_point_mask,
		           intrinsic_matrix, extrinsic_matrix, rendering_image_size);
	}
}

void
DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::GraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::RGBDImage& reference_image,
		const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix,
		float depth_scale
) const {
	FitToImage(warp_field, canonical_mesh, reference_image.color_, reference_image.depth_, reference_image_mask,
	           intrinsic_matrix, extrinsic_matrix,
	           depth_scale);

}

open3d::core::Tensor DeformableMeshToImageFitter::ComputeResiduals(
		open3d::t::geometry::PointCloud& rasterized_point_cloud,
		open3d::core::Tensor& residual_mask,
		const open3d::t::geometry::TriangleMesh& warped_mesh,
		const open3d::core::Tensor& pixel_face_indices,
		const open3d::core::Tensor& pixel_barycentric_coordinates,
		const open3d::core::Tensor& pixel_depths,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::PointCloud& reference_point_cloud,
		const open3d::core::Tensor& reference_point_mask,
		const open3d::core::Tensor& intrinsics
) const {

	o3c::SizeVector image_size = {reference_color_image.GetRows(), reference_color_image.GetCols()};

	auto vertex_normals = warped_mesh.GetVertexNormals();
	auto triangle_indices = warped_mesh.GetTriangleIndices();
	auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));

	auto rendered_normals =
			nnrt::rendering::functional::InterpolateVertexAttributes(
					pixel_face_indices, pixel_barycentric_coordinates, face_vertex_normals
			);

	o3c::Tensor identity_extrinsics = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));

	o3c::Tensor rasterized_points, rendered_point_mask;
	nnrt::geometry::functional::UnprojectDepthImageWithoutFiltering(
			rasterized_points, rendered_point_mask, pixel_depths, intrinsics, identity_extrinsics,
			1.0f, this->max_depth, false
	);

	rasterized_point_cloud = o3tg::PointCloud(rasterized_points);
	rasterized_point_cloud.SetPointNormals(rendered_normals.Reshape({-1, 3}));

	o3c::Tensor distances =
			geometry::functional::ComputePointToPlaneDistances(rasterized_point_cloud, reference_point_cloud);

	residual_mask = reference_point_mask.LogicalAnd(rendered_point_mask);

	core::functional::SetMaskedToValue(distances, residual_mask.LogicalNot(), 0.0f);

	return distances;
}


} // namespace nnrt::alignment