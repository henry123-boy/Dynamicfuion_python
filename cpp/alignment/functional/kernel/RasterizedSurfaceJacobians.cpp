//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/3/23.
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
#include "alignment/functional/kernel/RasterizedSurfaceJacobians.h"
#include "core/DeviceSelection.h"

namespace nnrt::alignment::functional::kernel {

void RasterizedSurfaceJacobians(
		open3d::core::Tensor& rendered_vertex_jacobians,
		open3d::core::Tensor& rendered_normal_jacobians,
		const open3d::core::Tensor& warped_vertex_positions,
		const open3d::core::Tensor& warped_triangle_indices,
		const open3d::core::Tensor& warped_vertex_normals,
		const open3d::core::Tensor& pixel_faces,
		const open3d::core::Tensor& pixel_barycentric_coordinates,
		const open3d::core::Tensor& ndc_intrinsics,
		bool perspective_corrected_barycentric_coordinates,
		bool compute_normal_jacobians
) {
	core::ExecuteOnDevice(
			warped_vertex_positions.GetDevice(),
			[&] {
				RasterizedSurfaceJacobians<open3d::core::Device::DeviceType::CPU>(
						rendered_vertex_jacobians, rendered_normal_jacobians, warped_vertex_positions,
						warped_triangle_indices, warped_vertex_normals, pixel_faces, pixel_barycentric_coordinates,
						ndc_intrinsics,
						perspective_corrected_barycentric_coordinates, true);
			},
			[&] {
				NNRT_IF_CUDA(
						RasterizedSurfaceJacobians<open3d::core::Device::DeviceType::CUDA>(
								rendered_vertex_jacobians, rendered_normal_jacobians, warped_vertex_positions,
								warped_triangle_indices, warped_vertex_normals, pixel_faces,
								pixel_barycentric_coordinates, ndc_intrinsics,
								perspective_corrected_barycentric_coordinates, true);
				);
			}
	);
}

} // namespace nnrt::alignment::functional::kernel
