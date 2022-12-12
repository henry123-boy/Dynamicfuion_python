//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/12/22.
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

// local
#include "EdgeShader.h"
#include "rendering/kernel/CoordinateSystemConversions.h"

namespace nnrt::rendering {

open3d::t::geometry::Image EdgeShader::ShadeMeshes(
		const open3d::core::Tensor& pixel_face_indices,
		const open3d::core::Tensor& pixel_depths,
		const open3d::core::Tensor& pixel_barycentric_coordinates,
		const open3d::core::Tensor& pixel_face_distances,
		const open3d::utility::optional<std::reference_wrapper<std::vector<open3d::t::geometry::TriangleMesh>>> meshes)
const {
	open3d::utility::LogError("Not implemented");
	return open3d::t::geometry::Image();
}

void EdgeShader::SetPixelLineWidth(float pixel_line_width) {
	this->_pixel_line_width = pixel_line_width;
	this->_ndc_width = kernel::ImageSpaceDistanceToNdc(pixel_line_width, this->rendered_image_size[0], this->rendered_image_size[1]);
}

} // nnrt::rendering