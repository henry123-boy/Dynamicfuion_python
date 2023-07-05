//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/5/23.
//  Copyright (c) 2023 Gregory Kramida
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

// local includes
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/DeviceSelection.h"

namespace nnrt::alignment::kernel {

void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
		open3d::core::Tensor& node_jacobians,
		open3d::core::Tensor& node_jacobian_ranges,
		open3d::core::Tensor& node_jacobian_pixel_indices,
		open3d::core::Tensor& node_pixel_jacobian_indices_jagged,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		const open3d::core::Tensor& pixel_node_jacobians
) {
	core::ExecuteOnDevice(
			node_pixel_jacobian_indices_jagged.GetDevice(),
			[&] {
				ConvertPixelVertexAnchorJacobiansToNodeJacobians<open3d::core::Device::DeviceType::CPU>(
						node_jacobians, node_jacobian_ranges, node_jacobian_pixel_indices,
						node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts, pixel_node_jacobians
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ConvertPixelVertexAnchorJacobiansToNodeJacobians<open3d::core::Device::DeviceType::CUDA>(
								node_jacobians, node_jacobian_ranges, node_jacobian_pixel_indices,
								node_pixel_jacobian_indices_jagged, node_pixel_jacobian_counts, pixel_node_jacobians
						);
				);
			}
	);
}

void ComputeDepthHessianApproximationBlocks_UnorderedNodePixels(
		open3d::core::Tensor& hessian_approximation_blocks,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		IterationMode mode
) {
	core::ExecuteOnDevice(
			pixel_jacobians.GetDevice(),
			[&] {
				ComputeDepthHessianApproximationBlocks_UnorderedNodePixels<open3d::core::Device::DeviceType::CPU>(
						hessian_approximation_blocks, pixel_jacobians, node_pixel_jacobian_indices, node_pixel_jacobian_counts, mode
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeDepthHessianApproximationBlocks_UnorderedNodePixels<open3d::core::Device::DeviceType::CUDA>(
								hessian_approximation_blocks, pixel_jacobians, node_pixel_jacobian_indices, node_pixel_jacobian_counts, mode
						);
				);
			}
	);
}

void ComputeNegativeDepthGradient_UnorderedNodePixels(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_vertex_anchor_count,
		IterationMode mode
) {
	core::ExecuteOnDevice(
			pixel_jacobians.GetDevice(),
			[&] {
				ComputeNegativeDepthGradient_UnorderedNodePixels<open3d::core::Device::DeviceType::CPU>(
						negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices,
						node_pixel_jacobian_counts, max_vertex_anchor_count, mode
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeNegativeDepthGradient_UnorderedNodePixels<open3d::core::Device::DeviceType::CUDA>(
								negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices,
								node_pixel_jacobian_counts, max_vertex_anchor_count, mode
						);
				);
			}
	);
}

void ComputeArapResiduals_FixedCoverageWeight(
		open3d::core::Tensor& edge_residuals,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& edge_layers,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& layer_node_weights,
		float arap_term_weight
) {
	core::ExecuteOnDevice(
			edges.GetDevice(),
			[&] {
				ComputeArapResiduals_FixedCoverageWeight<open3d::core::Device::DeviceType::CPU>(
						edge_residuals,
						edges,
						edge_layers,
						node_positions,
						node_translations,
						node_rotations,
						layer_node_weights,
						arap_term_weight
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeArapResiduals_FixedCoverageWeight<open3d::core::Device::DeviceType::CUDA>(
								edge_residuals,
								edges,
								edge_layers,
								node_positions,
								node_translations,
								node_rotations,
								layer_node_weights,
								arap_term_weight
						);
				);
			}
	);
}

void ComputeArapResiduals_VariableCoverageWeight(
		open3d::core::Tensor&       edge_residuals,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& node_rotations,
		float                       arap_term_weight
) {
	core::ExecuteOnDevice(
			edges.GetDevice(),
			[&] {
				ComputeArapResiduals_VariableCoverageWeight<open3d::core::Device::DeviceType::CPU>(
						edge_residuals,
						edges,
						node_positions,
						node_coverage_weights,
						node_translations,
						node_rotations,
						arap_term_weight
				);
			},
			[&] {
				NNRT_IF_CUDA(
						ComputeArapResiduals_VariableCoverageWeight<open3d::core::Device::DeviceType::CUDA>(
								edge_residuals,
								edges,
								node_positions,
								node_coverage_weights,
								node_translations,
								node_rotations,
								arap_term_weight
						);
				);
			}
	);
}


} // namespace nnrt::alignment::kernel