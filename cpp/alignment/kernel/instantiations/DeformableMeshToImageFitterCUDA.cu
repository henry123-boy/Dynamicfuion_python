//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/10/23.
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
#include "alignment/kernel/DeformableMeshToImageFitterImpl.h"
namespace nnrt::alignment::kernel {

template
void ConvertPixelVertexAnchorJacobiansToNodeJacobians<open3d::core::Device::DeviceType::CUDA>(
        open3d::core::Tensor& node_jacobians,
        open3d::core::Tensor& node_jacobian_ranges,
        open3d::core::Tensor& node_jacobian_pixel_indices,
        open3d::core::Tensor& node_pixel_jacobian_indices_jagged,
        const open3d::core::Tensor& node_pixel_counts,
        const open3d::core::Tensor& pixel_jacobians
);

template
void ComputeHessianApproximationBlocks_UnorderedNodePixels<open3d::core::Device::DeviceType::CUDA>(
        open3d::core::Tensor& workload_index,
        const open3d::core::Tensor& pixel_jacobians,
        const open3d::core::Tensor& node_pixel_jacobian_indices,
        const open3d::core::Tensor& node_pixel_jacobian_counts
);

template
void ComputeNegativeGradient_UnorderedNodePixels<open3d::core::Device::DeviceType::CUDA>(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex
);

} // namespace nnrt::alignment::kernel