//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/12/22.
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


// test framework
#include "tests/test_utils/test_main.hpp"
#include "tests/test_utils/test_utils.hpp"
#include "tests/test_utils/geometry.h"

// stdlib
#include <cmath>

// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <Eigen/Geometry>

// code being tested
#include "geometry/functional/Warping.h"

#define DegreesToRadians(angle_degrees) (angle_degrees * M_PI / 180.0)

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestWarpMeshWithNormals(const o3c::Device& device) {

	auto plane = test::GenerateXyPlane(1.0, std::make_tuple(0.f, 0.f, 0.f), 1, device);
	o3c::Tensor nodes = plane.GetVertexPositions();

	Eigen::Matrix<float, 9, 3, Eigen::RowMajor> nodes_eigen(nodes.ToFlatVector<float>().data());

	float mesh_rotation_angle = DegreesToRadians(90);
	auto rotation_eigen = Eigen::AngleAxisf(-mesh_rotation_angle, Eigen::Vector3f::UnitY()).toRotationMatrix();
	auto node_new_positions_eigen = (nodes_eigen * rotation_eigen.transpose()).eval(); // don't have to account for pivot, since that's (0,0,0)
	o3c::Tensor node_new_positions(node_new_positions_eigen.data(), {3, nodes.GetLength()}, o3c::Float32, device);
	node_new_positions = node_new_positions.Transpose(0, 1);
	o3c::Tensor node_translations = node_new_positions - nodes;
	o3c::Tensor single_node_rotation(rotation_eigen.transpose().eval().data(), {3, 3}, o3c::Float32, device);
	o3c::Tensor node_rotations = o3c::Tensor::Zeros({nodes.GetLength(), 3, 3}, o3c::Float32, device);
	for (int i_node = 0; i_node < nodes.GetLength(); i_node++) {
		node_rotations.Slice(0, i_node, i_node + 1) = single_node_rotation;
	}

	auto warped_plane = nnrt::geometry::functional::WarpTriangleMesh(plane, nodes, node_rotations, node_translations, 4, 0.1, false, 0);

	o3c::Tensor warped_plane_vertices_ground_truth(std::vector<float>{
			-0.f, -0.5f, -0.5f,
			-0.f, 0.5f, -0.5f,
			0.f, -0.5f, 0.5f,
			0.f, 0.5f, 0.5f,
			-0.f, 0.f, -0.5f,
			0.f, 0.f, 0.f,
			0.f, -0.5f, 0.f,
			0.f, 0.5f, 0.f,
			0.f, 0.f, 0.5f
	}, {nodes.GetLength(), 3}, o3c::Float32, device);

	o3c::Tensor warped_plane_normals_ground_truth(std::vector<float>{
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f,
			-1.f, 0.f, 0.f
	}, {nodes.GetLength(), 3}, o3c::Float32, device);

	REQUIRE(warped_plane.GetVertexPositions().AllClose(warped_plane_vertices_ground_truth,1e-5,1e-5));
	REQUIRE(warped_plane.GetVertexNormals().AllClose(warped_plane_normals_ground_truth, 1e-5,1e-7));

}

TEST_CASE("Test Warp Mesh With Normals CPU") {
	auto device = o3c::Device("CPU:0");
	TestWarpMeshWithNormals(device);
}

TEST_CASE("Test Warp Mesh With Normals CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestWarpMeshWithNormals(device);
}