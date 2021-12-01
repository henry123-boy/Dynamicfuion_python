//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/25/21.
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
#include "core/kernel/KdTree.h"
#include "core/DeviceSelection.h"
#include "core/DimensionCount.h"


namespace o3c = open3d::core;

namespace nnrt::core::kernel::kdtree {

void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points) {
	core::InferDeviceFromTensorAndExecute(
			points,
			[&] { BuildKdTreeIndex<o3c::Device::DeviceType::CPU>(index_data, points); },
			[&] { NNRT_IF_CUDA(BuildKdTreeIndex<o3c::Device::DeviceType::CPU>(index_data, points);); }
	);

}

void
FindKNearestKdTreePoints(open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
                         const open3d::core::Tensor& query_points,
                         int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points) {
	const int dimension_count = (int) kd_tree_points.GetShape(1);
	core::InferDeviceFromTensorAndExecute(
			kd_tree_points,
			[&] {
				FindKNearestKdTreePoints<o3c::Device::DeviceType::CPU>(
						closest_indices, squared_distances, query_points, k, index_data, kd_tree_points);
			},
			[&] {
				NNRT_IF_CUDA(
						FindKNearestKdTreePoints<o3c::Device::DeviceType::CPU>(
								closest_indices, squared_distances, query_points, k, index_data, kd_tree_points);
				);
			}
	);
}

} //  nnrt::core::kernel::kdtree
