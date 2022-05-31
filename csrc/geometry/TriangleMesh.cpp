//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/31/22.
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
#include "geometry/TriangleMesh.h"
#include "core/KdTree.h"

namespace nnrt::geometry {
TriangleMesh::TriangleMesh(const open3d::core::Tensor& points) {
	core::KdTree tree(points);
	open3d::core::Tensor nn_indices, nn_distances;
	tree.FindKNearestToPoints(nn_indices, nn_distances, points, 4, true);
	//TODO finish -- need to run a kernel on vertices going over their neighbors to compile triangles, crossing out neighbors from a list of atomics to avoid duplicate edges.
	// for this, need to sort neighbors at indices 1,2,and 3 by length first (index 0 will be edge start vertex), to prioritize shorter edges

	open3d::utility::LogError("Not fully implemented (see TODO in PointDownsamplingImpl.h)");

}
} // geometry