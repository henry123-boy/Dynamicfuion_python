//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/30/22.
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

// local includes
#include "alignment/functional/kernel/RasterizedSurfaceJacobiansImpl.h"

namespace nnrt::alignment::functional::kernel {
template
void RasterizedSurfaceJacobians<open3d::core::Device::DeviceType::CUDA>(
		open3d::core::Tensor& rendered_vertex_jacobians,
		utility::optional<std::reference_wrapper<open3d::core::Tensor>> rendered_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_positions,
		const open3d::core::Tensor& warped_triangle_indices,
		utility::optional<std::reference_wrapper<const open3d::core::Tensor>> warped_vertex_normals,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& pixel_barycentric_coordinates,
		const open3d::core::Tensor& ndc_intrinsics,
		bool perspective_corrected_barycentric_coordinates,
		bool compute_normal_jacobians
);
} // namespace nnrt::alignment::functional::kernel