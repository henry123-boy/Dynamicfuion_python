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
#pragma once

#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>


namespace nnrt::geometry {

void ComputeTriangleNormals(open3d::t::geometry::TriangleMesh& mesh, bool normalized = true);
void ComputeVertexNormals(open3d::t::geometry::TriangleMesh& mesh, bool normalized = true);
open3d::core::Tensor ComputeOrderedPointCloudNormals(const open3d::t::geometry::PointCloud& point_cloud, const open3d::core::SizeVector& source_image_size);
void NormalizeVectors3d(open3d::core::Tensor& vectors3d);


} // namespace nnrt::geometry