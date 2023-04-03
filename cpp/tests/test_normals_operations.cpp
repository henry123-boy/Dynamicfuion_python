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
// test framework
#include "test_main.hpp"
#include "tests/test_utils/test_utils.hpp"

// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/io/ImageIO.h>

// code being tested
#include "geometry/functional/NormalsOperations.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestComputeMeshTriangleNormals(const o3c::Device& device) {
	auto mesh_legacy = o3g::TriangleMesh::CreateSphere();
	o3tg::TriangleMesh mesh = o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);

	mesh_legacy->ComputeTriangleNormals(false);
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

void TestComputeMeshVertexNormals(const o3c::Device& device) {
	auto mesh_legacy = o3g::TriangleMesh::CreateSphere();
	o3tg::TriangleMesh mesh = o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);

	mesh_legacy->ComputeVertexNormals(false);
	nnrt::geometry::ComputeVertexNormals(mesh, false);

	auto vertex_normals = mesh.GetVertexNormals();
	auto vertex_normals_legacy = o3c::eigen_converter::EigenVector3dVectorToTensor(mesh_legacy->vertex_normals_, o3c::Float32, device);

	REQUIRE(vertex_normals.AllClose(vertex_normals_legacy));

	mesh_legacy = o3g::TriangleMesh::CreateSphere();
	mesh = o3tg::TriangleMesh::FromLegacy(*mesh_legacy, o3c::Float32, o3c::Int64, device);

	mesh_legacy->ComputeVertexNormals(true);
	nnrt::geometry::ComputeVertexNormals(mesh, true);

	vertex_normals = mesh.GetVertexNormals();
	vertex_normals_legacy = o3c::eigen_converter::EigenVector3dVectorToTensor(mesh_legacy->vertex_normals_, o3c::Float32, device);

	o3c::Tensor lhs = vertex_normals.To(o3c::Float64);
	o3c::Tensor rhs = vertex_normals_legacy.To(o3c::Float64);
	o3c::Tensor actual_error = (lhs - rhs).Abs();
	o3c::Tensor max_error = 5e-7 + 0.0 * rhs.Abs();
	o3c::Tensor comparison_result = actual_error <= max_error;
	auto bad_stuff_at = comparison_result.LogicalNot().NonZero().ToFlatVector<int64_t>();

	if (!bad_stuff_at.empty()) {
		int64_t bad_normal_index = comparison_result.LogicalNot().NonZero().ToFlatVector<int64_t>()[0];
		std::cout << vertex_normals.GetItem(o3c::TensorKey::Index(bad_normal_index)).ToString() << std::endl;
		std::cout << vertex_normals_legacy.GetItem(o3c::TensorKey::Index(bad_normal_index)).ToString() << std::endl;
		std::cout << bad_normal_index << std::endl;
	}

	REQUIRE(vertex_normals.AllClose(vertex_normals_legacy, 0.0, 5e-7));
}

TEST_CASE("Test Compute Mesh Vertex Normals CPU") {
	auto device = o3c::Device("CPU:0");
	TestComputeMeshVertexNormals(device);
}

TEST_CASE("Test Compute Mesh Vertex Normals CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestComputeMeshVertexNormals(device);
}

void TestComputeOrderedPointCloudNormals(const o3c::Device& device) {
	open3d::geometry::Image image;
	open3d::io::ReadImage(test::static_image_test_data_directory.ToString() + "/red_shorts_200_depth.png", image);
	o3tg::Image input_depth = o3tg::Image::FromLegacy(image);

	auto intrinsics_data = test::read_intrinsics(test::static_intrinsics_test_data_directory.ToString() + "/red_shorts_intrinsics.txt");
	// o3c::Device host("CPU:0");
	// o3c::Tensor intrinsics(intrinsics_data, {4, 4}, o3c::Float64, host);
	double fx = intrinsics_data[0];
	double fy = intrinsics_data[5];
	double cx = intrinsics_data[2];
	double cy = intrinsics_data[6];

	open3d::camera::PinholeCameraIntrinsic intrinsics_legacy(input_depth.GetCols(),input_depth.GetRows(), fx, fy, cx, cy);

	auto point_cloud_legacy = open3d::geometry::PointCloud::CreateFromDepthImage(image, intrinsics_legacy);
	auto point_cloud = o3tg::PointCloud::FromLegacy(*point_cloud_legacy, o3c::Float32, device);

	o3c::Tensor normals = nnrt::geometry::ComputeOrderedPointCloudNormals(point_cloud, {input_depth.GetRows(), input_depth.GetCols()});

	o3c::Tensor ground_truth_normals = o3c::Tensor::Load(test::static_array_test_data_directory.ToString() + "/red_shorts_200_normals.npy");

	REQUIRE(normals.AllClose(ground_truth_normals));

}

TEST_CASE("Test Ordered Point Cloud Normals CPU") {
	auto device = o3c::Device("CPU:0");
	TestComputeMeshVertexNormals(device);
}

TEST_CASE("Test Ordered Point Cloud Normals  CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestComputeMeshVertexNormals(device);
}