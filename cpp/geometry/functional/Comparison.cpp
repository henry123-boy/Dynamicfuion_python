//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/4/21.
//  Copyright (c) 2021 Gregory Kramida
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
#include "Comparison.h"
#include "geometry/functional/kernel/Comparison.h"

using namespace open3d;
using namespace open3d::t::geometry;


namespace nnrt::geometry {

open3d::core::Tensor
ComputePointToPlaneDistances(const TriangleMesh& mesh1, const TriangleMesh& mesh2) {

	if (mesh1.GetDevice() != mesh2.GetDevice()) {
		utility::LogError("Devices for two meshes need to match. Got: {} and {}.", mesh1.GetDevice().ToString(), mesh2.GetDevice().ToString());
	}
	if (!mesh1.HasVertexNormals()) {
		utility::LogError("Mesh1 needs to have vertex normals defined.");
	}
	const core::Tensor& vertices1 = mesh1.GetVertexPositions();
	const core::Tensor& vertices2 = mesh2.GetVertexPositions();
	const core::Tensor& normals1 = mesh1.GetVertexNormals();
	if (vertices1.GetLength() != vertices2.GetLength()) {
		utility::LogError("Meshes need to have matching number of vertices. Got: {} and {}.", vertices1.GetLength(), vertices2.GetLength());
	}

	core::Tensor distances;
	kernel::comparison::ComputePointToPlaneDistances(distances, normals1, vertices1, vertices2);
	return distances;
}

open3d::core::Tensor
ComputePointToPlaneDistances(const TriangleMesh& mesh, const PointCloud& point_cloud) {
	if (mesh.GetDevice() != point_cloud.GetDevice()) {
		utility::LogError("Devices for the mesh and point cloud need to match. Got: {} and {}.",
						  mesh.GetDevice().ToString(), point_cloud.GetDevice().ToString());
	}
	if (!mesh.HasVertexNormals()) {
		utility::LogError("Mesh needs to have vertex normals defined.");
	}
	const core::Tensor& vertex_normals = mesh.GetVertexNormals();
	const core::Tensor& vertex_positions = mesh.GetVertexPositions();
	const core::Tensor& point_positions = point_cloud.GetPointPositions();

	if (vertex_positions.GetLength() != point_positions.GetLength()) {
		utility::LogError("Mesh vertex count has to match the point count in the point cloud. Got: {} and {}.",
		                  vertex_positions.GetLength(), point_positions.GetLength());
	}

	core::Tensor distances;
	kernel::comparison::ComputePointToPlaneDistances(distances, vertex_normals, vertex_positions, point_positions);
	return distances;
}

open3d::core::Tensor ComputePointToPlaneDistances(const PointCloud& point_cloud1, const PointCloud& point_cloud2) {
	if (point_cloud1.GetDevice() != point_cloud2.GetDevice()) {
		utility::LogError("Devices for the mesh and point cloud need to match. Got: {} and {}.",
		                  point_cloud1.GetDevice().ToString(), point_cloud2.GetDevice().ToString());
	}
	if (!point_cloud1.HasPointNormals()) {
		utility::LogError("Mesh needs to have vertex normals defined.");
	}
	const core::Tensor& point_normals1 = point_cloud1.GetPointNormals();
	const core::Tensor& point_positions1 = point_cloud1.GetPointPositions();
	const core::Tensor& point_positions2 = point_cloud2.GetPointPositions();

	if (point_positions1.GetLength() != point_positions2.GetLength()) {
		utility::LogError("Point count in point cloud 1 has to match the point count in the point cloud 2. Got: {} and {}.",
		                  point_positions1.GetLength(), point_positions2.GetLength());
	}

	core::Tensor distances;
	kernel::comparison::ComputePointToPlaneDistances(distances, point_normals1, point_positions1, point_positions2);
	return distances;
}
} //namespace nnrt::geometry
