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

#include "core/kernel/KdTreeUtils.h"
#include "core/kernel/KdTreePointCloud.h"
#include "core/KeyValuePair.h"
#include "core/DeviceHeap.h"
#include "core/PlatformIndependence.h"


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::kdtree {

namespace {
template<open3d::core::Device::DeviceType TDeviceType, typename TPoint>
NNRT_DEVICE_WHEN_CUDACC
inline void Swap(KdTreePointCloudNode<TPoint>* node1, KdTreePointCloudNode<TPoint>* node2) {
	TPoint tmp_point = node1->point;
	node1->point = node2->point;
	node2->point = tmp_point;
}

template<open3d::core::Device::DeviceType TDeviceType, typename TPoint>
NNRT_DEVICE_WHEN_CUDACC
inline KdTreePointCloudNode<TPoint>*
FindMedian(KdTreePointCloudNode<TPoint>* nodes, int32_t range_start, int32_t range_end, int32_t i_dimension) {
	KdTreePointCloudNode<TPoint>* range_start_node = nodes + range_start;
	KdTreePointCloudNode<TPoint>* range_end_node = nodes + range_end;
	if (range_end <= range_start) return nullptr;
	if (range_end == range_start + 1) return range_start_node;

	KdTreePointCloudNode<TPoint>* cursor;
	KdTreePointCloudNode<TPoint>* swap_target;
	KdTreePointCloudNode<TPoint>* middle = range_start_node + (range_end_node - range_start_node) / 2;


	// if (end - start > 3) {
	float pivot;

	// find median using quicksort-like algorithm
	while (true) {
		KdTreePointCloudNode<TPoint>* last = range_end_node - 1;
		// get pivot (coordinate in the proper dimension) from the median candidate at the middle
		pivot = middle->point.coeff(i_dimension);
		// push the middle element toward the end
		Swap<TDeviceType>(middle, last);
		// traverse the range from start to finish using the cursor. Initialize reference to the range start.
		for (swap_target = cursor = range_start_node; cursor < last; cursor++) {
			// at every iteration, compare the cursor to the pivot. If it precedes the pivot, swap it with the reference,
			// thereby pushing the preceding value towards the start of the range.
			if (cursor->point.coeff(i_dimension) < pivot) {
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
		if (swap_target->point.coeff(i_dimension) ==
		    middle->point.coeff(i_dimension)) {
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


template<open3d::core::Device::DeviceType TDeviceType, typename TPoint>
NNRT_DEVICE_WHEN_CUDACC
inline void FindTreeNodeAndSetUpChildRanges(RangeNode<TPoint>* range_nodes, RangeNode<TPoint>* range_nodes_end, RangeNode<TPoint>* range_node,
                                            KdTreePointCloudNode<TPoint>* nodes, uint8_t i_dimension) {

	KdTreePointCloudNode<TPoint>* median_node = FindMedian<TDeviceType>(nodes, range_node->range_start, range_node->range_end, i_dimension);
	range_node->node = median_node;
	if (median_node == nullptr) {
		return;
	}
	median_node->i_dimension = i_dimension;
	auto median_node_index = static_cast<int32_t>(median_node - nodes);

	auto parent_index = range_node - range_nodes;

	RangeNode<TPoint>* left_child = range_nodes + 2 * parent_index + 1;
	RangeNode<TPoint>* right_child = range_nodes + 2 * parent_index + 2;

	if (left_child < range_nodes_end) {
		left_child->node = nullptr;
		left_child->range_start = range_node->range_start;
		left_child->range_end = median_node_index;
		if (right_child < range_nodes_end) {
			right_child->node = nullptr;
			right_child->range_start = median_node_index + 1;
			right_child->range_end = range_node->range_end;
		}
	}
}
} // namespace

template<open3d::core::Device::DeviceType TDeviceType, typename TPoint, typename TMakePoint>
void BuildKdTreePointCloud_Generic(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root, TMakePoint&& make_point) {
	const int64_t point_count = points.GetLength();

	const auto dimension_count = (int32_t) points.GetShape(1);
	o3gk::NDArrayIndexer point_indexer(points, 1);
	auto* nodes = reinterpret_cast<KdTreePointCloudNode<TPoint>*>(index_data.GetDataPtr());

	int tree_level_count;
	int64_t secondary_index_length = FindBalancedTreeIndexLength(point_count, tree_level_count);

	open3d::core::Blob secondary_index(secondary_index_length * static_cast<int64_t>(sizeof(RangeNode<TPoint>)), index_data.GetDevice());

	auto* range_nodes = reinterpret_cast<RangeNode<TPoint>*>(secondary_index.GetDataPtr());
	RangeNode<TPoint>* range_nodes_end = range_nodes + secondary_index_length;


	// index points linearly at first
	open3d::core::ParallelFor(
			index_data.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				KdTreePointCloudNode<TPoint>& node = nodes[workload_idx];
				node.point = make_point(point_indexer.template GetDataPtr<float>(workload_idx));
				node.i_dimension = 0;
				node.left_child = nullptr;
				node.right_child = nullptr;
			}
	);


	open3d::core::ParallelFor(
			index_data.GetDevice(), 1,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				RangeNode<TPoint>& range_node = range_nodes[workload_idx];
				range_nodes[workload_idx].node = nodes;
				range_node.range_start = 0;
				range_node.range_end = static_cast<int32_t>(point_count);
			}
	);
	uint8_t i_dimension = 0;
	// build tree by splitting each node down the median along a different dimension at each iteration
	for (int64_t range_start_index = 0, range_length = 1;
	     range_start_index < point_count;
	     range_start_index += range_length, range_length *= 2) {

		open3d::core::ParallelFor(
				index_data.GetDevice(), range_length,
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					RangeNode<TPoint>* range_node = range_nodes + range_start_index + workload_idx;
					FindTreeNodeAndSetUpChildRanges<TDeviceType, TPoint>(range_nodes, range_nodes_end, range_node, nodes, i_dimension);
				}
		);
		i_dimension = (i_dimension + 1) % dimension_count;
	}

	o3c::Tensor root_index_tensor({1}, o3c::Dtype::Int32, points.GetDevice());
	auto* root_index_at = root_index_tensor.GetDataPtr<int32_t>();

	// convert from index-based tree to direct / pointer-based tree to facilitate simple subsequent searches & insertion
	open3d::core::ParallelFor(
			index_data.GetDevice(), secondary_index_length - IntPower(2, tree_level_count - 1),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				RangeNode<TPoint>& range_node = range_nodes[workload_idx];
				if (workload_idx == 0) {
					*root_index_at = static_cast<int32_t>(range_node.node - nodes);
				}
				if (range_node.node != nullptr) {
					RangeNode<TPoint>& left_child_candidate = range_nodes[2 * workload_idx + 1];
					RangeNode<TPoint>& right_child_candidate = range_nodes[2 * workload_idx + 2];
					range_node.node->left_child = left_child_candidate.node;
					range_node.node->right_child = right_child_candidate.node;
				}
			}
	);

	*root = nodes + *(root_index_tensor.To(o3c::Device("CPU:0")).template GetDataPtr<int32_t>());
}

template<open3d::core::Device::DeviceType TDeviceType>
void BuildKdTreePointCloud(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root) {
	auto dimension_count = (int32_t) points.GetShape(1);
	switch (dimension_count) {
		case 1:
			BuildKdTreePointCloud_Generic<TDeviceType, Eigen::Vector<float, 1>>(
					index_data, points, root,
					[]NNRT_DEVICE_WHEN_CUDACC(float* point_data) { return Eigen::Vector<float, 1>(point_data[0]); }
			);
			break;
		case 2:
			BuildKdTreePointCloud_Generic<TDeviceType, Eigen::Vector2f>(
					index_data, points, root,
					[]NNRT_DEVICE_WHEN_CUDACC(float* point_data) { return Eigen::Vector2f(point_data[0], point_data[1]); }
			);
			break;
		case 3:
			BuildKdTreePointCloud_Generic<TDeviceType, Eigen::Vector3f>(
					index_data, points, root,
					[]NNRT_DEVICE_WHEN_CUDACC(float* point_data) { return Eigen::Vector3f(point_data[0], point_data[1], point_data[2]); }
			);
			break;
		default:
			BuildKdTreePointCloud_Generic<TDeviceType, Eigen::Vector<float, Eigen::Dynamic>>(
					index_data, points, root, [dimension_count]NNRT_DEVICE_WHEN_CUDACC(float* point_data) {
						Eigen::Vector<float, Eigen::Dynamic> point(dimension_count);
						for (int i_value = 0; i_value < dimension_count; i_value++) {
							point << point_data[i_value];
						}
						return point;
					}
			);
			break;
	}
}


} // nnrt::core::kernel::kdtree