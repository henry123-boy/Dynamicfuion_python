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

#include <cfloat>

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>

#include "core/kernel/KdTreeUtilities.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/kernel/KdTree.h"
#include "core/KeyValuePair.h"
#include "core/DeviceHeap.h"
#include "core/PlatformIndependence.h"


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::kdtree {

namespace {
template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void Swap(KdTreeNode* node1, KdTreeNode* node2) {
	int32_t tmp_index = node1->point_index;
	node1->point_index = node2->point_index;
	node2->point_index = tmp_index;
}

template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline KdTreeNode*
FindMedian(KdTreeNode* nodes, int32_t range_start, int32_t range_end, int32_t i_dimension, const o3gk::NDArrayIndexer& point_indexer) {
	KdTreeNode* range_start_node = nodes + range_start;
	KdTreeNode* range_end_node = nodes + range_end;
	if (range_end <= range_start) return nullptr;
	if (range_end == range_start + 1) return range_start_node;

	KdTreeNode* cursor;
	KdTreeNode* swap_target;
	KdTreeNode* middle = range_start_node + (range_end_node - range_start_node) / 2;

	auto coordinate = [&point_indexer, &i_dimension](KdTreeNode* node) {
		return point_indexer.template GetDataPtr<float>(node->point_index)[i_dimension];
	};

	// if (end - start > 3) {
	float pivot;

	// find median using quicksort-like algorithm
	while (true) {
		KdTreeNode* last = range_end_node - 1;
		// get pivot (coordinate in the proper dimension) from the median candidate at the middle
		pivot = coordinate(middle);
		// push the middle element toward the end
		Swap<TDeviceType>(middle, last);
		// traverse the range from start to finish using the cursor. Initialize reference to the range start.
		for (swap_target = cursor = range_start_node; cursor < last; cursor++) {
			// at every iteration, compare the cursor to the pivot. If it precedes the pivot, swap it with the reference,
			// thereby pushing the preceding value towards the start of the range.
			if (coordinate(cursor) < pivot) {
				if (cursor != swap_target) {
					Swap<TDeviceType>(cursor, swap_target);
				}
				swap_target++;
			}
		}
		// push swap_target to the back of the range, bringing the original median candidate ("middle") to the swap_target location
		Swap<TDeviceType>(swap_target, last);

		// at some point, with the whole lower half of the range sorted, reference is going to end up at the true median,
		// in which case return that node (because it means that exactly half the nodes are below the median candidate).
		// We do a coordinate- instead of pointer comparison here, because there might be duplicate values around the median
		if (coordinate(swap_target) == coordinate(middle)) {
			return middle;
		}

		// Reference now holds the old median, while median holds the end-of-range.
		// Here we decide which side of the remaining range we need to sort, left or right of the median.
		if (swap_target > middle) {
			range_end_node = swap_target;
		} else {
			range_start_node = swap_target;
		}
	}
}

template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindTreeNodeAndSetUpChildRanges(RangeNode* range_nodes, RangeNode* range_nodes_end, RangeNode* range_node,
                                            KdTreeNode* nodes, uint8_t i_dimension,
                                            const o3gk::NDArrayIndexer& point_indexer) {

	KdTreeNode* median_node = FindMedian<TDeviceType>(nodes, range_node->range_start, range_node->range_end, i_dimension, point_indexer);
	if (median_node == nullptr) {
		return;
	}
	median_node->i_split_dimension = i_dimension;
	range_node->node = *median_node;

	auto median_node_index = static_cast<int32_t>(median_node - nodes);

	auto parent_index = range_node - range_nodes;

	RangeNode* left_child = range_nodes + 2 * parent_index + 1;
	RangeNode* right_child = range_nodes + 2 * parent_index + 2;

	if (left_child < range_nodes_end) {
		left_child->range_start = range_node->range_start;
		left_child->range_end = median_node_index;
		if (right_child < range_nodes_end) {
			right_child->range_start = median_node_index + 1;
			right_child->range_end = range_node->range_end;
		}
	}
}
} // namespace

// __DEBUG
// #define DEBUG_ST
#ifdef DEBUG_ST
namespace cpu_launcher_st {
template<typename func_t>
void ParallelFor(o3c::Device& device, int64_t n, const func_t& func) {
	for (int64_t i = 0; i < n; ++i) {
		func(i);
	}
}
} // namespace cpu_launcher_st
#endif

template<open3d::core::Device::DeviceType TDeviceType>
void BuildKdTreeIndex(open3d::core::Blob& index_data, int64_t index_length, const open3d::core::Tensor& points) {
	const int64_t point_count = points.GetLength();

	const auto dimension_count = (int32_t) points.GetShape(1);
	o3gk::NDArrayIndexer point_indexer(points, 1);
	auto* nodes = reinterpret_cast<KdTreeNode*>(index_data.GetDataPtr());

	open3d::core::Blob secondary_index(index_length * static_cast<int64_t>(sizeof(RangeNode)), index_data.GetDevice());

	auto* range_nodes = reinterpret_cast<RangeNode*>(secondary_index.GetDataPtr());
	RangeNode* range_nodes_end = range_nodes + index_length;


	// index points linearly at first using the nodes
	open3d::core::ParallelFor(
			index_data.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				KdTreeNode& node = nodes[workload_idx];
				node.point_index = static_cast<int32_t>(workload_idx);
				node.i_split_dimension = 0;
			}
	);

	// set up the range nodes so they all point "nowhere" at first
	open3d::core::ParallelFor(
			index_data.GetDevice(), index_length,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				RangeNode& range_node = range_nodes[workload_idx];
				range_node.node.Clear();
			}
	);


	open3d::core::ParallelFor(
			index_data.GetDevice(), 1,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				RangeNode& range_node = range_nodes[workload_idx];
				range_node.range_start = 0;
				range_node.range_end = static_cast<int32_t>(point_count);
			}
	);
	uint8_t i_dimension = 0;
	// build tree by splitting each node down the median along a different dimension at each tree level/iteration of this loop
	for (int64_t range_start_index = 0, range_length = 1;
	     range_start_index < point_count;
	     range_start_index += range_length, range_length *= 2) {

		open3d::core::ParallelFor(
				index_data.GetDevice(), range_length,
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					RangeNode* range_node = range_nodes + range_start_index + workload_idx;
					FindTreeNodeAndSetUpChildRanges<TDeviceType>(range_nodes, range_nodes_end, range_node, nodes, i_dimension,
					                                             point_indexer);
				}
		);
		i_dimension = (i_dimension + 1) % dimension_count;
	}

	// copy over the nodes from range nodes
	open3d::core::ParallelFor(
			index_data.GetDevice(), index_length,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				nodes[workload_idx] = range_nodes[workload_idx].node;
			}
	);
}


} // nnrt::core::kernel::kdtree