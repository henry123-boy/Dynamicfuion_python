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
#include "PointToPlaneDistance.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt::geometry{
open3d::core::Tensor
ComputePointToPlaneDistances(open3d::t::geometry::TriangleMesh& mesh1, open3d::t::geometry::TriangleMesh& mesh2) {

	if(mesh1.GetDevice() != mesh2.GetDevice()){
		utility::LogError("Devices for two meshes need to match. Got: {} and {}.", mesh1.GetDevice().ToString(), mesh2.GetDevice().ToString());
	}
	if(!mesh1.HasVertexNormals()){
		utility::LogError("Mesh1 needs to have vertex normals defined.");
	}
	auto vertices1 = mesh1.GetVertices();
	auto vertices2 = mesh2.GetVertices();
	auto normals1 = mesh1.GetVertexNormals();



}
} //namespace nnrt::geometry
