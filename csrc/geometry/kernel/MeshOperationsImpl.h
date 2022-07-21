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

#include "geometry/kernel/MeshOperations.h"
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::geometry::kernel::mesh {

template<open3d::core::Device::DeviceType TDevice>
void ComputeTriangleNormals(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                            const open3d::core::Tensor& triangle_indices) {
	auto device = vertex_positions.GetDevice();
	triangle_normals = o3c::Tensor({triangle_indices.GetLength(), 3}, vertex_positions.GetDtype(), device);
	o3gk::NDArrayIndexer vertex_indexer(vertex_positions, 1);
	o3gk::NDArrayIndexer triangle_indexer(triangle_indices, 1);
	o3c::ParallelFor(
			device, triangle_indices.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				Eigen::Map<Eigen::Matrix<int64_t,1,3>> triangles(triangle_indexer.template GetDataPtr<int64_t>(workload_idx));

			}
	);


}

} // nnrt::geometry::kernel::mesh