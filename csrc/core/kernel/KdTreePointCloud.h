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

template<typename TPoint>
struct KdTreePointCloudNode {
	TPoint point;
	uint8_t i_dimension;
	KdTreePointCloudNode* left_child;
	KdTreePointCloudNode* right_child;
};

template<typename TPoint>
struct RangeNode {
	KdTreePointCloudNode<TPoint>* node;
	int32_t range_start;
	int32_t range_end;
};

enum SearchStrategy {
	RECURSIVE, ITERATIVE
};

size_t GetNodeByteCount(const open3d::core::Tensor& points);

template<typename TPoint>
open3d::core::Blob PointCloudDataToHost(const open3d::core::Blob& index_data, int point_count, int dimension_count);


void PointCloudDataToHost_CUDA(open3d::core::Blob& node_data_cpu, const open3d::core::Blob& node_data, int point_count, int dimension_count);

void BuildKdTreePointCloud(open3d::core::Blob& node_data, const open3d::core::Tensor& points, void** root);

template<open3d::core::Device::DeviceType DeviceType>
void BuildKdTreePointCloud(open3d::core::Blob& node_data, const open3d::core::Tensor& points, void** root);

template<SearchStrategy TSearchStrategy>
void FindKNearestKdTreePointCloudPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
                                        const open3d::core::Tensor& query_points, int32_t k, const void* root, int dimension_count);

template<open3d::core::Device::DeviceType DeviceType, SearchStrategy TSearchStrategy>
void FindKNearestKdTreePointCloudPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
                                        const open3d::core::Tensor& query_points, int32_t k, const void* root, int dimension_count);

void
GenerateKdTreePointCloudDiagram(std::string& point, const open3d::core::Blob& node_data, const void* root, int point_count,
                                int dimension_count, int digit_length);

} //  nnrt::core::kernel::kdtree