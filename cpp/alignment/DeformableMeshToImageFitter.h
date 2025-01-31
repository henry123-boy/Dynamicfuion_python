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
#pragma once


// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/RGBDImage.h>

// local
#include "geometry/HierarchicalGraphWarpField.h"
#include "alignment/IterationMode.h"

namespace nnrt::alignment {

class DeformableMeshToImageFitter {
public:
	//TODO: we need here to have various terminations conditions that can work together or be turned on and off, i.e. probably
	// use a separate struct holding data that dictates to the DeformableMeshToImageFitter how to facilitate this behavior (and is stored locally)
	explicit DeformableMeshToImageFitter(
			int max_iteration_count = 100,
			std::vector<IterationMode> iteration_mode_sequence = {IterationMode::ALL},
			float minimal_update_threshold = 1e-6f,
			bool use_perspective_correction = true,
			float max_depth = 10.f,
			bool use_tukey_penalty_for_data_term = false,
			float tukey_penalty_cutoff_cm = 0.01f,
			float preconditioning_dampening_factor = 0.0f,
			float arap_term_weight = 200.0f,
			bool use_huber_penalty_for_arap_term = false,
			float huber_penalty_constant = 0.0001
	);

	/**
	 * \brief
	 * \param warp_field
	 * \param canonical_mesh
	 * \param reference_image
	 * \param intrinsic_matrix
	 * \param extrinsic_matrix -- Note: has nothing to do with the reference RGBD image, which is assumed to have identity extrinsics (rel. to camera)
	 * \param depth_scale
	 * \param depth_max
	 */
	void FitToImage(
			nnrt::geometry::HierarchicalGraphWarpField& warp_field,
			const open3d::t::geometry::TriangleMesh& canonical_mesh,
			const open3d::t::geometry::RGBDImage& reference_image,
			const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
			const open3d::core::Tensor& intrinsic_matrix,
			const open3d::core::Tensor& extrinsic_matrix,
			float depth_scale
	)
	const;

	void FitToImage(
			nnrt::geometry::HierarchicalGraphWarpField& warp_field,
			const open3d::t::geometry::TriangleMesh& canonical_mesh,
			const open3d::t::geometry::Image& reference_color_image,
			const open3d::t::geometry::Image& reference_depth_image,
			const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
			const open3d::core::Tensor& intrinsic_matrix,
			const open3d::core::Tensor& extrinsic_matrix,
			float depth_scale
	)
	const;

	void FitToImage(
			nnrt::geometry::HierarchicalGraphWarpField& warp_field,
			const open3d::t::geometry::TriangleMesh& canonical_mesh,
			const open3d::t::geometry::Image& reference_color_image,
			const open3d::t::geometry::PointCloud& reference_point_cloud,
			const open3d::core::Tensor& reference_point_mask,
			const open3d::core::Tensor& intrinsic_matrix,
			const open3d::core::Tensor& extrinsic_matrix,
			const open3d::core::SizeVector& rendering_image_size
	)
	const;


private:
	int max_iteration_count;
	std::vector<IterationMode> iteration_mode_sequence;
	float min_update_threshold;
	float max_depth;
	bool use_perspective_correction;
	bool use_tukey_penalty_for_depth_term;
	float tukey_penalty_cutoff_cm;
	float levenberg_marquart_factor;
	float arap_term_weight;
	bool use_huber_penalty_for_arap_term;
	float huber_penalty_constant;


	open3d::core::Tensor ComputeDepthResiduals(
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
	)
	const;

	open3d::core::Tensor ComputeEdgeResiduals(
			geometry::HierarchicalGraphWarpField& warp_field
	) const;

};


} // namespace nnrt::alignment
