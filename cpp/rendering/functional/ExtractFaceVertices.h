//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/19/22.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>

// local includes

namespace nnrt::rendering::functional{

open3d::core::Tensor GetMeshFaceVerticesNdc(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor> GetMeshNdcFaceVerticesAndClipMask(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> GetMeshFaceVerticesNdcAndNormalsNdcAndClipMask(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);

} // namespace nnrt::rendering::functional