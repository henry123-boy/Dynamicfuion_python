//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/22/22.
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
#include "test_main.hpp"

#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/t/geometry/TriangleMesh.h>

#include "geometry/NormalsOperations.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestComputeMeshTriangleNormals(const o3c::Device& device) {
	auto mesh_legacy = o3g::TriangleMesh::CreateSphere();
	mesh_legacy->ComputeTriangleNormals(false);
	o3tg::TriangleMesh mesh = o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);
	nnrt::geometry::ComputeTriangleNormals(mesh, false);

	auto triangle_normals = mesh.GetTriangleNormals();
	auto triangle_normals_legacy = o3c::eigen_converter::EigenVector3dVectorToTensor(mesh_legacy->triangle_normals_, o3c::Float32, device);

	REQUIRE(triangle_normals.AllClose(triangle_normals_legacy));

	mesh_legacy->ComputeTriangleNormals(true);
	nnrt::geometry::ComputeTriangleNormals(mesh, true);

	triangle_normals = mesh.GetTriangleNormals();
	triangle_normals_legacy = o3c::eigen_converter::EigenVector3dVectorToTensor(mesh_legacy->triangle_normals_, o3c::Float32, device);

	REQUIRE(triangle_normals.AllClose(triangle_normals_legacy));
}

TEST_CASE("Test Compute Mesh Triangle Normals CPU") {
	auto device = o3c::Device("CPU:0");
	TestComputeMeshTriangleNormals(device);
}

TEST_CASE("Test Compute Mesh Triangle Normals CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestComputeMeshTriangleNormals(device);
}

void TestComputeMeshVertexNormals(const o3c::Device& device){
	auto mesh_legacy = o3g::TriangleMesh::CreateSphere();
	mesh_legacy->ComputeTriangleNormals(true);
	mesh_legacy->ComputeVertexNormals(false);
	o3tg::TriangleMesh mesh = o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);
	nnrt::geometry::ComputeTriangleNormals(mesh, true);
	nnrt::geometry::ComputeVertexNormals(mesh, false);

	auto vertex_normals = mesh.GetVertexNormals();
	auto vertex_normals_legacy = o3c::eigen_converter::EigenVector3dVectorToTensor(mesh_legacy->vertex_normals_, o3c::Float32, device);

	std::cout << vertex_normals.ToString() << std::endl << "Expected: " << vertex_normals_legacy.ToString() << std::endl;

	o3c::Tensor lhs = vertex_normals.To(o3c::Float64);
	o3c::Tensor rhs = vertex_normals_legacy.To(o3c::Float64);
	o3c::Tensor actual_error = (lhs - rhs).Abs();
	// std::cout << actual_error.GetItem(o3c::TensorKey::Index(700)).ToString() << std::endl;

	o3c::Tensor max_error = 1e-7 + 0.0 * rhs.Abs();
	auto close_tensor = actual_error <= max_error;

	// std::cout << close_tensor.GetItem(o3c::TensorKey::Index(700)).ToString() << std::endl;

	std::cout << close_tensor.ToString() << std::endl;

	REQUIRE(vertex_normals.AllClose(vertex_normals_legacy,0.0,1e-7));
}

TEST_CASE("Test Compute Mesh Vertex Normals CPU") {
	auto device = o3c::Device("CPU:0");
	TestComputeMeshVertexNormals(device);
}

TEST_CASE("Test Compute Mesh Vertex Normals CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestComputeMeshVertexNormals(device);
}