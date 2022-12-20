//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/5/21.
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
#pragma once

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Core>

namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt::geometry::functional::kernel::comparison {
template<open3d::core::Device::DeviceType TDeviceType>
void ComputePointToPlaneDistances(open3d::core::Tensor& distances,
                                  const open3d::core::Tensor& normals1,
                                  const open3d::core::Tensor& vertices1,
                                  const open3d::core::Tensor& vertices2) {
	const int64_t point_count = vertices1.GetLength();
	distances = o3c::Tensor::Zeros({point_count}, o3c::Dtype::Float32, vertices1.GetDevice());
	NDArrayIndexer distances_indexer(distances, 1);
	NDArrayIndexer normals1_indexer(vertices1, 1);
	NDArrayIndexer vertices1_indexer(vertices1, 1);
	NDArrayIndexer vertices2_indexer(vertices2, 1);

	open3d::core::ParallelFor(
			vertices1.GetDevice(),
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto vertex1_normal_data = normals1_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> vertex1_normal(vertex1_normal_data);
				auto vertex1_position_data = vertices1_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> vertex1(vertex1_position_data);
				auto vertex2_position_data = vertices2_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> vertex2(vertex2_position_data);
				auto* distance_data = distances_indexer.template GetDataPtr<float>(workload_idx);
				*distance_data = vertex1_normal.template dot(vertex1 - vertex2);
			}
	);
}
} // namespace nnrt::geometry::functional::kernel::comparison