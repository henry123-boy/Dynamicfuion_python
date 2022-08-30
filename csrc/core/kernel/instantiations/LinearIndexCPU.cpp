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
#include <open3d/core/ParallelFor.h>

#include "core/heap/CPU/DeviceHeapCPU.h"
#include "core/kernel/LinearIndexImpl.h"

namespace nnrt::core::kernel::linear_index {

template
void FindKNearestKdTreePoints<open3d::core::Device::DeviceType::CPU, NeighborTrackingStrategy::PLAIN>(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
		const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points
);

template
void FindKNearestKdTreePoints<open3d::core::Device::DeviceType::CPU, NeighborTrackingStrategy::PRIORITY_QUEUE>(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
		const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points
);

} // nnrt::core::kernel::linear_index