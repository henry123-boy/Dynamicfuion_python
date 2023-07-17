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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/t/io/TriangleMeshIO.h>
#include <open3d/t/io/ImageIO.h>
#include <open3d/geometry/Geometry.h>

// local includes
#include "tests/test_utils/fitter_testing.h"
#include "tests/test_utils/test_utils.hpp"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tio = open3d::t::io;
namespace o3tg = open3d::t::geometry;

namespace test{


std::tuple<o3tg::TriangleMesh, o3tg::TriangleMesh> ReadAndTransformTwoMeshes(
		const std::string& mesh_name_1, const std::string& mesh_name_2,
		const o3c::Device& device, const o3c::Tensor& mesh_transform
) {
	o3tg::TriangleMesh source_mesh, target_mesh;


	o3tio::ReadTriangleMesh(generated_mesh_test_data_directory.ToString() + "/" + mesh_name_1 + ".ply", source_mesh);
	o3tio::ReadTriangleMesh(generated_mesh_test_data_directory.ToString() + "/" + mesh_name_2 + ".ply", target_mesh);


	source_mesh = source_mesh.To(device);
	target_mesh = target_mesh.To(device);

	o3c::Tensor extrinsic_matrix = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));

	// move 1.2 units away from the camera
	target_mesh = target_mesh.Transform(mesh_transform);
	source_mesh = source_mesh.Transform(mesh_transform);
	return std::make_tuple(source_mesh, target_mesh);
}


} // namespace test