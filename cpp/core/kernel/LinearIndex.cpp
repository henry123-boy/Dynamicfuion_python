//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/8/22.
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
#include "LinearIndex.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::core::kernel::linear_index {
template<NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(
		o3c::Tensor& nearest_neighbor_indices, o3c::Tensor& squared_distances,
		const o3c::Tensor& query_points, int32_t k, const o3c::Tensor& indexed_points) {
	core::InferDeviceFromEntityAndExecute(
			indexed_points,
			[&] {
				FindKNearestKdTreePoints<o3c::Device::DeviceType::CPU, TTrackingStrategy>(
						nearest_neighbor_indices, squared_distances, query_points, k, indexed_points);
			},
			[&] {
				NNRT_IF_CUDA(
						FindKNearestKdTreePoints<o3c::Device::DeviceType::CUDA, TTrackingStrategy>(
								nearest_neighbor_indices, squared_distances, query_points, k, indexed_points);
				);
			}
	);
}

template
void FindKNearestKdTreePoints<NeighborTrackingStrategy::PLAIN>(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
		const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points
);
template
void FindKNearestKdTreePoints<NeighborTrackingStrategy::PRIORITY_QUEUE>(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
		const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points
);

} // namespace nnrt::core::kernel::linear_index