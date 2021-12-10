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
#include "core/DimensionCount.h"

namespace nnrt::core::kernel::kdtree {

struct KdTreeNode {
	int32_t index;
	union {
		KdTreeNode* range_start;
		KdTreeNode* left_child;
	};
	union {
		KdTreeNode* range_end;
		KdTreeNode* right_child;
	};
};

void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points);

template<open3d::core::Device::DeviceType DeviceType>
void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points);

void FindKNearestKdTreePoints(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
		int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points
);

template<open3d::core::Device::DeviceType DeviceType>
void FindKNearestKdTreePoints(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
		int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points
);

void GetNodeIndices(open3d::core::Tensor& indices, const open3d::core::Blob& index_data, int64_t node_count);

template<open3d::core::Device::DeviceType DeviceType>
void GetNodeIndices(open3d::core::Tensor& indices, const open3d::core::Blob& index_data, int64_t node_count);

} //  nnrt::core::kernel::kdtree