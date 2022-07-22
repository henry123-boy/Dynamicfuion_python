//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/21/22.
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

#include <open3d/utility/Logging.h>

#include "geometry/NormalsOperations.h"
#include "geometry/kernel/NormalsOperations.h"


namespace o3u = open3d::utility;
namespace o3c = open3d::core;

namespace nnrt::geometry {

void CheckMeshVerticesAndTriangles(const open3d::t::geometry::TriangleMesh& mesh){
	if (!mesh.HasVertexPositions() || !mesh.HasTriangleIndices()) {
		o3u::LogError("Mesh needs to have both vertex positions and triangle indices to compute triangle normals. Vertex positions: {}."
		              " Triangle indices: {}.",
		              mesh.HasVertexPositions() ? "present" : "absent", mesh.HasTriangleIndices() ? "present" : "absent");
	}
}

void ComputeTriangleNormals(open3d::t::geometry::TriangleMesh& mesh, bool normalized /* = true */) {
	CheckMeshVerticesAndTriangles(mesh);
	o3c::Tensor vertex_positions = mesh.GetVertexPositions();
	o3c::Tensor triangle_indices = mesh.GetTriangleIndices();
	o3c::Tensor triangle_normals;
	kernel::mesh::ComputeTriangleNormals(triangle_normals, vertex_positions, triangle_indices);
	if(normalized){
		NormalizeVectors3d(triangle_normals);
	}
	mesh.SetTriangleNormals(triangle_normals);
}

void NormalizeVectors3d(open3d::core::Tensor& vectors3d) {
	kernel::mesh::NormalizeVectors3d(vectors3d);
}

void ComputeVertexNormals(open3d::t::geometry::TriangleMesh& mesh, bool normalized) {
	CheckMeshVerticesAndTriangles(mesh);
	if(!mesh.HasTriangleNormals()){
		ComputeTriangleNormals(mesh, false);
	}
	o3c::Tensor vertex_positions = mesh.GetVertexPositions();
	o3c::Tensor triangle_indices = mesh.GetTriangleIndices();
	o3c::Tensor triangle_normals = mesh.GetTriangleNormals();
	o3c::Tensor vertex_normals;

	kernel::mesh::ComputeVertexNormals(vertex_normals, vertex_positions, triangle_indices, triangle_normals);

	if(normalized){
		NormalizeVectors3d(vertex_normals);
	}
	mesh.SetVertexNormals(vertex_normals);

}

} // namespace nnrt::geometry