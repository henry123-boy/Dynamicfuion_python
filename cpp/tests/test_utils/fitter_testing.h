//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/11/23.
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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/t/geometry/TriangleMesh.h>

// local includes

namespace test{

std::tuple<open3d::t::geometry::TriangleMesh, open3d::t::geometry::TriangleMesh> ReadAndTransformTwoMeshes(
		const std::string& mesh_name_1, const std::string& mesh_name_2,
		const open3d::core::Device& device, const open3d::core::Tensor& mesh_transform
);

} // namespace test