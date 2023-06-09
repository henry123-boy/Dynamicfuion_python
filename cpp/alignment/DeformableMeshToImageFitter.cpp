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

#include <utility>

// local
#include "core/functional/Masking.h"
#include "core/linalg/SolveBlockDiagonalCholesky.h"
#include "alignment/DeformableMeshToImageFitter.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/PointToPlaneDistances.h"
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "alignment/functional/WarpedSurfaceJacobians.h"
#include "alignment/functional/RasterizedSurfaceJacobians.h"
#include "alignment/functional/PixelVertexAnchorJacobians.h"
#include "alignment/functional/AssociateFacesWithAnchors.h"
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/linalg/Rodrigues.h"
#include "alignment/functional/ArapJacobian.h"
#include "alignment/functional/ArapHessian.h"

namespace o3tio = open3d::t::io;

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;


#define DEBUG_HANDPICKED_DATA_PIXEL_REGION

namespace nnrt::alignment {

DeformableMeshToImageFitter::DeformableMeshToImageFitter(
		int max_iteration_count /* = 100*/,
		std::vector<IterationMode> iteration_mode_sequence /* = {IterationMode::ALL}*/,
		float minimal_update_threshold /* = 1e-6f*/,
		bool use_perspective_correction /* = true*/,
		float max_depth /* = 10.0f*/,
		bool use_tukey_penalty_for_data_term /* = false*/,
		float tukey_penalty_cutoff_cm /* = 0.01f*/,
		float preconditioning_dampening_factor /* = 0.0f*/,
		float arap_term_weight /*= 0.1f*/,
		bool use_huber_penalty_for_arap_term /* = false*/,
		float huber_penalty_constant /* = 0.0001*/
) : max_iteration_count(max_iteration_count),
    iteration_mode_sequence(std::move(iteration_mode_sequence)),
    min_update_threshold(minimal_update_threshold),
    max_depth(max_depth),
    use_perspective_correction(use_perspective_correction),
    use_tukey_penalty_for_depth_term(use_tukey_penalty_for_data_term),
    tukey_penalty_cutoff_cm(tukey_penalty_cutoff_cm),
    preconditioning_dampening_factor(preconditioning_dampening_factor),
    arap_term_weight(arap_term_weight),
    use_huber_penalty_for_arap_term(use_huber_penalty_for_arap_term),
    huber_penalty_constant(huber_penalty_constant) {
	if (preconditioning_dampening_factor < 0.0f || preconditioning_dampening_factor > 1.0f) {
		utility::LogError("`preconditioning_dampening_factor` should be a small non-negative value between 0 and 1. Got: {}",
		                  preconditioning_dampening_factor);
	}
}

void DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::HierarchicalGraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::Image& reference_color_image,
		const open3d::t::geometry::PointCloud& reference_point_cloud,
		const open3d::core::Tensor& reference_point_mask,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix,
		const open3d::core::SizeVector& rendering_image_size
) const {
	//TODO: make parameter EUCLIDEAN/SHORTEST_PATH, optionally set in constructor
	auto [warp_anchors, warp_anchor_weights] =
			warp_field.PrecomputeAnchorsAndWeights(canonical_mesh, true);

	int iteration = 0;
	float maximum_update = std::numeric_limits<float>::max();

	auto [ndc_intrinsic_matrix, ndc_xy_range] =
			rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, rendering_image_size);

	auto [face_node_anchors, face_node_anchor_counts] =
			functional::AssociateFacesWithAnchors(canonical_mesh.GetTriangleIndices(), warp_anchors);

	bool use_regularization_term = warp_field.GetEdges().GetLength() > 0;

	//TODO: fix second termination condition -- probably, better to use overall energy or residual sum.
	while (iteration < max_iteration_count && maximum_update > min_update_threshold) {
		IterationMode current_mode = this->iteration_mode_sequence[iteration % this->iteration_mode_sequence.size()];

		o3tg::TriangleMesh
				warped_mesh = warp_field.WarpMesh(canonical_mesh, warp_anchors, warp_anchor_weights, true, extrinsic_matrix);

		auto [extracted_face_vertices, clipped_face_mask] =
				nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(
						warped_mesh, intrinsic_matrix, rendering_image_size, 0.0, 10.0
				);

		//TODO: add an obvious optional optimization in the case perspective-correct barycentrics are used: output the
		// distorted barycentric coordinates and reuse them, instead of recomputing them, in RasterizedVertexAndNormalJacobians
		std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> fragments =
				nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, rendering_image_size, 0.5f, 1, -1, -1,
				                                       this->use_perspective_correction, false, true);
		auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] = fragments;

		// compute residuals r, retain rasterized points & global mask relevant for energy function being minimized
		o3tg::PointCloud rasterized_point_cloud;
		o3c::Tensor residual_mask;
		// [PX]; PX = W * H
		o3c::Tensor depth_residuals =
				this->ComputeDepthResiduals(
						rasterized_point_cloud, residual_mask, warped_mesh, pixel_face_indices,
						pixel_barycentric_coordinates, pixel_depths,
						reference_color_image, reference_point_cloud, reference_point_mask,
						intrinsic_matrix
				);


		o3c::Tensor edge_residuals;
		if (use_regularization_term) {
			edge_residuals = this->ComputeEdgeResiduals(warp_field);
		}

#ifdef DEBUG_HANDPICKED_DATA_PIXEL_REGION
		int region_width = 10;
		int region_height = 9;
		int pixel_x_start = 45;
		int pixel_x_end = pixel_x_start + region_width;
		int pixel_y_start = 22;
		int pixel_y_end = pixel_y_start + region_height;
		o3c::Tensor region_depth_residuals = depth_residuals.Reshape(rendering_image_size).Slice(0, pixel_y_start, pixel_y_end).Contiguous()
		                                                    .Slice(1, pixel_x_start, pixel_x_end).Contiguous();
#endif

		//TODO: revise termination conditions to check the residual magnitudes / energy somehow
		o3c::Tensor warped_vertex_position_jacobians, warped_vertex_normal_jacobians;

		if (current_mode == IterationMode::ALL || current_mode == IterationMode::ROTATION_ONLY) {
			// compute warped vertex and normal jacobians wrt. delta rotations and jacobians
			// [V X A/V X 4], [V X A/V X 3]
			std::tie(warped_vertex_position_jacobians, warped_vertex_normal_jacobians) =
					functional::WarpedSurfaceJacobians(canonical_mesh, warp_field, warp_anchors, warp_anchor_weights, true);
		} else {
			warped_vertex_position_jacobians = warp_anchor_weights;
		}

		// compute rasterized vertex and normal jacobians
		// [W x H x 3 x 9], [W x H x 30]; 30 = 3x9, 1x3
		//TODO: probably better to explicitly separate the rasterized_vertex_normal_jacobians ([W x H x 3x9]) from the barycentrics ([W x H x 1x3])
		// for clarity's sake. Also, it makes sense to be explicit that it's not the full "rasterized_vertex_normal_jacobians"
		// (missing the barycentrics part), or at least insert some comments about this
		auto [rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians] =
				functional::RasterizedSurfaceJacobians(
						warped_mesh, pixel_face_indices,
						pixel_barycentric_coordinates,
						ndc_intrinsic_matrix,
						this->use_perspective_correction
				);


		// compute J_d, i.e. sparse Jacobian at every pixel w.r.t. every node delta
		o3c::Tensor point_map_vectors = rasterized_point_cloud.GetPointPositions() - reference_point_cloud.GetPointPositions();
		o3c::Tensor rasterized_normals = rasterized_point_cloud.GetPointNormals();

		auto [pixel_jacobians, pixel_node_jacobian_counts,
				node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts] =
				functional::PixelVertexAnchorJacobiansAndNodeAssociations(
						rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
						warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
						point_map_vectors, rasterized_normals, residual_mask, pixel_face_indices,
						face_node_anchors, face_node_anchor_counts, warp_field.node_positions.GetLength(),
						use_tukey_penalty_for_depth_term, tukey_penalty_cutoff_cm, current_mode
				);
#ifdef DEBUG_HANDPICKED_DATA_PIXEL_REGION
		int jacobian_stride = current_mode == IterationMode::ALL ? 6 : 3;
		int max_face_anchor_count = 3 * static_cast<int>(warped_vertex_position_jacobians.GetShape(1));
		o3c::Tensor region_pixel_jacobians =
				pixel_jacobians.Reshape({rendering_image_size[0], rendering_image_size[1], max_face_anchor_count, jacobian_stride})
				               .Slice(0, pixel_y_start, pixel_y_end).Contiguous().Slice(1, pixel_x_start, pixel_x_end).Contiguous();
#endif


		// compute (J_d^T)J_d, i.e. hessian approximation for the data term
		open3d::core::Tensor hessian_blocks_depth_diagonal;
		kernel::ComputeDepthHessianApproximationBlocks_UnorderedNodePixels(
				hessian_blocks_depth_diagonal, pixel_jacobians,
				node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts, current_mode
		);


		// compute -(J_d^T)r_d (negative gradient for the data term)
		o3c::Tensor negative_gradient_depth;
		int max_anchor_count_per_vertex = static_cast<int32_t>(warp_anchors.GetShape(1));
		kernel::ComputeNegativeDepthGradient_UnorderedNodePixels(
				negative_gradient_depth, depth_residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices_jagged,
				node_pixel_jacobian_counts, max_anchor_count_per_vertex, current_mode
		);

		open3d::core::Tensor motion_updates;
		if (use_regularization_term) {
			int64_t first_layer_node_count = warp_field.GetRegularizationLevel(0).node_indices.GetLength();

			// compute J_r, i.e. ARAP term sparse jacobian
			o3c::Tensor edge_jacobians = functional::ComputeDenseArapEdgeJacobians(warp_field, this->arap_term_weight);

			// compute sparse H_r
			core::linalg::BlockSparseArrowheadMatrix hessian_approximation =
					functional::ComputeArapBlockSparseHessianApproximation(warp_field.GetEdges(), edge_jacobians, first_layer_node_count,
					                                                       warp_field.GetNodePositions(false).GetLength());

			hessian_approximation.diagonal_blocks += hessian_blocks_depth_diagonal;
			if (preconditioning_dampening_factor > 0.0) {
				kernel::PreconditionDiagonalBlocks(hessian_approximation.diagonal_blocks, preconditioning_dampening_factor);
			}


		} else {
			if (preconditioning_dampening_factor > 0.0) {
				kernel::PreconditionDiagonalBlocks(hessian_blocks_depth_diagonal, preconditioning_dampening_factor);
			}
			core::linalg::SolveBlockDiagonalCholesky(motion_updates, hessian_blocks_depth_diagonal, negative_gradient_depth);
		}


		o3c::Tensor rotation_matrix_updates;
		switch (current_mode) {
			case ALL: motion_updates = motion_updates.Reshape({motion_updates.GetShape(0) / 6, 6});
				// convert rotation axis-angle vectors to matrices
				rotation_matrix_updates = core::linalg::AxisAngleVectorsToMatricesRodrigues(motion_updates.Slice(1, 0, 3).Contiguous());
				// apply motion updates
				warp_field.TranslateNodes(motion_updates.Slice(1, 3, 6), true);
				warp_field.RotateNodes(rotation_matrix_updates, true);
				break;
			case TRANSLATION_ONLY: motion_updates = motion_updates.Reshape({motion_updates.GetShape(0) / 3, 3});
				warp_field.TranslateNodes(motion_updates, true);
				break;
			case ROTATION_ONLY: motion_updates = motion_updates.Reshape({motion_updates.GetShape(0) / 3, 3});
				rotation_matrix_updates = core::linalg::AxisAngleVectorsToMatricesRodrigues(motion_updates);
				warp_field.RotateNodes(rotation_matrix_updates, true);
				break;
		}
		iteration++;
	}
}

void
DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::HierarchicalGraphWarpField& warp_field,
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

	FitToImage(warp_field, canonical_mesh, reference_color_image, point_cloud, final_reference_point_mask,
	           intrinsic_matrix, extrinsic_matrix, rendering_image_size);

}

void
DeformableMeshToImageFitter::FitToImage(
		nnrt::geometry::HierarchicalGraphWarpField& warp_field,
		const open3d::t::geometry::TriangleMesh& canonical_mesh,
		const open3d::t::geometry::RGBDImage& reference_image,
		const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::Tensor& extrinsic_matrix,
		float depth_scale
) const {
	FitToImage(warp_field, canonical_mesh, reference_image.color_, reference_image.depth_, reference_image_mask,
	           intrinsic_matrix, extrinsic_matrix, depth_scale);

}

open3d::core::Tensor DeformableMeshToImageFitter::ComputeDepthResiduals(
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

	if (!warped_mesh.HasVertexNormals()) {
		utility::LogError("Input warped mesh needs to have normals defined; it doesn't.");
	}

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

	if (this->use_tukey_penalty_for_depth_term) {
		float c = this->tukey_penalty_cutoff_cm;
		float c_squared_over_six = (c * c / 6.f);
		o3c::Tensor c_squared_over_six_tensor(std::vector<float>{c_squared_over_six}, {1}, o3c::Float32, triangle_indices.GetDevice());
		o3c::Tensor distances_over_c = distances / c;
		o3c::Tensor left_operand = 1.f - (distances_over_c * distances_over_c);
		o3c::Tensor residuals = c_squared_over_six * (1.f - left_operand * left_operand * left_operand);
		o3c::Tensor locations_of_residuals_below_c = distances.Le(c);
		residuals.SetItem(o3c::TensorKey::IndexTensor(locations_of_residuals_below_c), c_squared_over_six_tensor);
		return residuals;
	} else {
		return distances;
	}
}

open3d::core::Tensor DeformableMeshToImageFitter::ComputeEdgeResiduals(geometry::HierarchicalGraphWarpField& warp_field) const {
	const o3c::Tensor& node_positions = warp_field.GetNodePositions(true);
	const o3c::Tensor& node_translations = warp_field.GetNodeTranslations(true);
	const o3c::Tensor& node_rotations = warp_field.GetNodeRotations(true);

	const o3c::Tensor& edges = warp_field.GetEdges();
	o3c::Tensor edge_residuals;

	switch (warp_field.warp_node_coverage_computation_method) {
		case geometry::WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE: {
			const o3c::Tensor& edge_layer_indices = warp_field.GetEdgeLayerIndices();
			const o3c::Tensor& layer_decimation_radii = warp_field.GetLayerDecimationRadii();
			kernel::ComputeArapResiduals_FixedCoverageWeight(
					edge_residuals,
					edges,
					edge_layer_indices,
					node_positions,
					node_translations,
					node_rotations,
					layer_decimation_radii,
					this->arap_term_weight
			);
		}
			break;

		case geometry::WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE: {
			const o3c::Tensor& node_coverage_weights = warp_field.GetNodeCoverageWeights(true);
			kernel::ComputeArapResiduals_VariableCoverageWeight(
					edge_residuals,
					edges,
					node_positions,
					node_coverage_weights,
					node_translations,
					node_rotations,
					this->arap_term_weight
			);
		}
			break;
		default: utility::LogError("Unsupported warp node coverage computation method: {}", warp_field.warp_node_coverage_computation_method);
			break;
	}

	if (use_huber_penalty_for_arap_term) {
		float delta = this->huber_penalty_constant;
		float half_of_delta_squared = 0.5f * delta * delta;
		o3c::Tensor half_of_z_squared_tensor(std::vector<float>{half_of_delta_squared}, {1}, o3c::Float32, edge_residuals.GetDevice());
		o3c::TensorKey locations_of_residuals_above_delta = o3c::TensorKey::IndexTensor(edge_residuals.Ge(delta));
		o3c::Tensor edge_residuals_huber = 0.5 * edge_residuals * edge_residuals;
		o3c::Tensor edge_residuals_huber_above_delta = edge_residuals.GetItem(locations_of_residuals_above_delta);
		edge_residuals_huber_above_delta.Abs_();
		edge_residuals_huber_above_delta -= half_of_z_squared_tensor;
		edge_residuals_huber.SetItem(locations_of_residuals_above_delta, edge_residuals_huber_above_delta);
		return edge_residuals_huber;
	} else {
		return edge_residuals;
	}
}


} // namespace nnrt::alignment