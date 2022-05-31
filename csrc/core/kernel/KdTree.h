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
#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/core/Blob.h>
#include "core/DimensionCount.h"

namespace nnrt::core::kernel::kdtree {

enum NeighborTrackingStrategy {
	PLAIN, PRIORITY_QUEUE
};

void BuildKdTreeIndex(open3d::core::Blob& index_data, int64_t index_length, const open3d::core::Tensor& points);

template<open3d::core::Device::DeviceType DeviceType>
void BuildKdTreeIndex(open3d::core::Blob& index_data, int64_t index_length, const open3d::core::Tensor& points);

template<NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
                              open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
                              const open3d::core::Tensor& reference_points);

template<open3d::core::Device::DeviceType DeviceType, NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
                              open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
                              const open3d::core::Tensor& reference_points);


void GenerateTreeDiagram(std::string& diagram, const open3d::core::Blob& index_data, int index_length,
                         const open3d::core::Tensor& kd_tree_points, int digit_length);

} //  nnrt::core::kernel::kdtree