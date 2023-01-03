//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
#include "geometry/GraphWarpField.h"

namespace nnrt::alignment::functional {
std::tuple<open3d::core::Tensor, open3d::core::Tensor> RenderedVertexAndNormalJacobians(
	const open3d::t::geometry::TriangleMesh& warped_mesh, const open3d::core::Tensor& pixel_faces,
	const open3d::core::Tensor& barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsics,
	bool perspective_corrected_barycentric_coordinates
);
} // namespace nnrt::alignment::functional