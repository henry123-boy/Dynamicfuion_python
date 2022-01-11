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

enum NeighborTrackingStrategy {
	PLAIN, PRIORITY_QUEUE
};

size_t GetNodeByteCount(const open3d::core::Tensor& points) {
	switch (points.GetShape(2)) {
		case 1:
			return sizeof(KdTreePointCloudNode<Eigen::Vector<float, 1>>);
		case 2:
			return sizeof(KdTreePointCloudNode<Eigen::Vector2f>);
		case 3:
			return sizeof(KdTreePointCloudNode<Eigen::Vector3f>);
		default:
			return sizeof(KdTreePointCloudNode<Eigen::Vector<float, Eigen::Dynamic>>);
	}
}

open3d::core::Blob BlobToDevice(const open3d::core::Blob& index_data, int64_t byte_count, const open3d::core::Device& device);

open3d::core::Blob IndexDataToHost(const open3d::core::Blob& index_data, int point_count);

void IndexDataToHost_CUDA(open3d::core::Blob& index_data_cpu, const open3d::core::Blob& index_data, int point_count);

void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root);

template<open3d::core::Device::DeviceType DeviceType>
void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root);

template<SearchStrategy TSearchStrategy, NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                              const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points, const void* root);

template<open3d::core::Device::DeviceType DeviceType, SearchStrategy TSearchStrategy, NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                              const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& kd_tree_points, const void* root);

void GenerateTreeDiagram(std::string& diagram, const open3d::core::Blob& index_data, const void* root, const open3d::core::Tensor& kd_tree_points,
                         int digit_length);

} //  nnrt::core::kernel::kdtree