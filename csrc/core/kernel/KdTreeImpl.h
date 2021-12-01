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

#include "core/KeyValuePair.h"
#include "core/DeviceHeap.h"
#include "core/kernel/KdTree.h"
#include "core/PlatformIndependence.h"


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::kdtree {

namespace {
template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void Swap(KdTreeNode* node1, KdTreeNode* node2) {
	uint32_t tmp_index = node1->index;
	node1->index = node2->index;
	node2->index = tmp_index;
}

template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline KdTreeNode* FindMedian(KdTreeNode* start, KdTreeNode* end, int32_t i_dimension, const o3gk::NDArrayIndexer& point_indexer) {
	if (end < start) return nullptr;
	if (end == start + 1) return start;

	KdTreeNode* cursor;
	KdTreeNode* reference;
	KdTreeNode* median_candidate = start + (end - start) / 2;

	float pivot;

	// find median using quicksort-like algorithm
	while (true) {
		// get pivot (coordinate in the proper dimension) from the median
		pivot = point_indexer.template GetDataPtr<float>(median_candidate->index)[i_dimension];
		// push current median candidate to the end of the range
		Swap<TDeviceType>(median_candidate, end - 1);
		// traverse the range from start to finish using the cursor. Initialize reference to the range start.
		for (reference = cursor = start; cursor < end; cursor++) {
			// at every iteration, compare the cursor to the pivot. If it precedes the pivot, swap it with the reference,
			// thereby pushing the preceding value towards the start of the range (before the pivot).
			if (point_indexer.template GetDataPtr<float>(cursor->index)[i_dimension] < pivot) {
				if (cursor != reference) {
					Swap<TDeviceType>(cursor, reference);
				}
				reference++;
			}
		}
		// push reference to the back of the range, bringing the original median candidate to the reference location
		Swap<TDeviceType>(reference, end - 1);

		// at some point, with the whole lower half of the range sorted, reference is going to end up at the true median,
		// in which case return that node (because it means that exactly half the nodes are below the median candidate)
		if (point_indexer.template GetDataPtr<float>(reference->index)[i_dimension] ==
		    point_indexer.template GetDataPtr<float>(median_candidate->index)[i_dimension]) {
			return median_candidate;
		}

		// Reference now holds the old median, while median holds the end-of-range.
		// Here we decide which side of the remaining range we need to sort, left or right of the median.
		if (reference > median_candidate) {
			end = reference;
		} else {
			start = reference;
		}

	}

}

template<open3d::core::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindTreeNodeAndSetUpChildRanges(KdTreeNode* nodes, KdTreeNode* nodes_end, KdTreeNode* destination,
                                            int32_t i_dimension, const o3gk::NDArrayIndexer& point_indexer) {
	KdTreeNode* median = FindMedian<TDeviceType>(destination->range_start, destination->range_end, i_dimension, point_indexer);
	// Place median at the destination
	Swap<TDeviceType>(destination, median);
	auto parent_index = nodes - destination;

	KdTreeNode* left_child = nodes + 2 * parent_index + 1;
	KdTreeNode* right_child = nodes + 2 * parent_index + 2;

	if (left_child < nodes_end) {
		left_child->range_start = destination + 1;
		left_child->range_end = median;
		if (right_child < nodes_end) {
			right_child->range_start = median;
			right_child->range_end = destination->range_end;
		}
	}
}
} // namespace


template<open3d::core::Device::DeviceType TDeviceType>
void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points) {
	const int64_t point_count = points.GetLength();
	const auto dimension_count = (int32_t) points.GetShape(1);
	o3gk::NDArrayIndexer point_indexer(points, 1);
	auto* nodes = reinterpret_cast<KdTreeNode*>(index_data.GetDataPtr());
	KdTreeNode* nodes_end = nodes + point_count;


#if defined(__CUDACC__)
	namespace launcher = o3c::kernel::cuda_launcher;
#else
	namespace launcher = o3c::kernel::cpu_launcher;
#endif

	// index points linearly at first
	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				KdTreeNode& node = nodes[workload_idx];
				node.index = static_cast<uint32_t>(workload_idx);
			}
	);

	launcher::ParallelFor(
			1,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				KdTreeNode* node = nodes;
				node->range_start = node;
				node->range_end = nodes + point_count;
			}
	);
	int32_t i_dimension = 0;
	// build tree by splitting each node down the median along a different dimension at each iteration
	for (int64_t range_start_index = 0, range_length = 1;
	     range_start_index < point_count;
	     range_start_index += range_length, range_length *= 2) {

		launcher::ParallelFor(
				std::min(range_length, point_count-range_start_index),
				[=] OPEN3D_DEVICE(int64_t workload_idx) {
					KdTreeNode* node = nodes + range_start_index + workload_idx;
					FindTreeNodeAndSetUpChildRanges<TDeviceType>(nodes, nodes_end, node, i_dimension, point_indexer);
				}
		);
		i_dimension = (i_dimension + 1) % dimension_count;
	}
	// convert from index-based tree to direct / pointer-based tree to speed up lookup & facilitate easy subsequent insertion
	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				KdTreeNode& node = nodes[workload_idx];

				KdTreeNode* left_child_candidate = nodes + 2 * workload_idx + 1;
				KdTreeNode* right_child_candidate = nodes + 2 * workload_idx + 2;
				if (left_child_candidate < nodes_end) {
					node.left_child = left_child_candidate;
					if (right_child_candidate < nodes_end) {
						node.right_child = right_child_candidate;
					} else {
						node.right_child = nullptr;
					}
				} else {
					node.left_child = nullptr;
				}
			}
	);

}


namespace {

typedef DistanceIndexPair<float, int32_t> KdDistanceIndexPair;
typedef decltype(core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>) NeighborCompare;
template<open3d::core::Device::DeviceType TDeviceType>
using NearestNeighborHeap = core::DeviceHeap<TDeviceType, KdDistanceIndexPair, NeighborCompare>;

template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
void FindKnnInKdSubtree(const KdTreeNode* node, NearestNeighborHeap<TDeviceType> nearest_neighbor_heap,
                        const TPointVector& query_point,
                        const o3gk::NDArrayIndexer& kd_tree_point_indexer, const int i_dimension,
                        const int dimension_count, TMakePointVector&& make_point_vector) {

	if (node == nullptr) {
		return;
	}
	float max_knn_distance = nearest_neighbor_heap.Head().distance;
	auto node_point = make_point_vector(kd_tree_point_indexer.template GetDataPtr<float>(node->index));

	float node_distance = (node_point - query_point).squaredNorm();
	if (max_knn_distance > node_distance) {
		nearest_neighbor_heap.Insert(KdDistanceIndexPair{node_distance, node->index});
	}

	float node_value = node_point.coeff(i_dimension);

	if (query_point[i_dimension] < node_value) {
		if (query_point[i_dimension] - max_knn_distance < node_value) {
			FindKnnInKdSubtree<TDeviceType>(node->left_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
		if (query_point[i_dimension] + max_knn_distance > node_value) {
			FindKnnInKdSubtree<TDeviceType>(node->right_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
	} else {
		if (query_point[i_dimension] + max_knn_distance < node_value) {
			FindKnnInKdSubtree<TDeviceType>(node->right_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
		if (query_point[i_dimension] - max_knn_distance > node_value) {
			FindKnnInKdSubtree<TDeviceType>(node->left_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
	}
}


template<open3d::core::Device::DeviceType TDeviceType, typename TMakePointVector>
inline void
FindKNearestKdTreePoints_Generic(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
		int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points, TMakePointVector&& make_point_vector
) {
	auto query_point_count = query_points.GetLength();
	auto kd_tree_point_count = kd_tree_points.GetLength();

	auto dimension_count = (int32_t) kd_tree_points.GetShape(1);


	auto* root_node = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	auto* node_end = root_node + kd_tree_point_count;
	o3gk::NDArrayIndexer kd_tree_point_indexer(kd_tree_points, 1);
	o3gk::NDArrayIndexer query_point_indexer(query_points, 1);

	closest_indices = open3d::core::Tensor({query_point_count, k}, o3c::UInt32, query_points.GetDevice());
	squared_distances = open3d::core::Tensor({query_point_count, k}, o3c::Float32, query_points.GetDevice());
	o3gk::NDArrayIndexer closest_indices_indexer(closest_indices, 1);
	o3gk::NDArrayIndexer squared_distance_indexer(squared_distances, 1);

#if defined(__CUDACC__)
	namespace launcher = o3c::kernel::cuda_launcher;
#else
	namespace launcher = o3c::kernel::cpu_launcher;
#endif

	launcher::ParallelFor(
			query_point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto query_point = make_point_vector(query_point_indexer.template GetDataPtr<float>(workload_idx));

				auto* nearest_neighbor_data = new DistanceIndexPair<float, int32_t>[k];

				core::DeviceHeap<TDeviceType, DistanceIndexPair<float, int32_t>,
						decltype(core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>)>
						nearest_neighbor_heap(
						nearest_neighbor_data, k, core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>
				);

				const KdTreeNode* node = node_end - 1;
				for (int i_node = 0; i_node < k && node >= root_node; i_node++, node--) {
					auto kd_tree_point = make_point_vector(kd_tree_point_indexer.template GetDataPtr<float>(node->index));
					float distance = (kd_tree_point - query_point).squaredNorm();
					nearest_neighbor_heap.Insert(KdDistanceIndexPair{distance, node->index});
				}


				FindKnnInKdSubtree<TDeviceType>(root_node, nearest_neighbor_heap, query_point, kd_tree_point_indexer, 0, dimension_count, make_point_vector);
				auto* indices_for_query_point = closest_indices_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto* distances_for_query_point = squared_distance_indexer.template GetDataPtr<float>(workload_idx);

				int i_neighbor = 0;
				while (!nearest_neighbor_heap.Empty()) {
					KdDistanceIndexPair pair = nearest_neighbor_heap.Pop();
					indices_for_query_point[i_neighbor] = pair.value;
					distances_for_query_point[i_neighbor] = pair.key;
					i_neighbor++;
				}
				delete[] nearest_neighbor_data;
				for (; i_neighbor < k; i_neighbor++) {
					indices_for_query_point[i_neighbor] = -1;
					distances_for_query_point[i_neighbor] = FLT_MAX;
				}
			}
	);
}
} // namespace

template<open3d::core::Device::DeviceType TDeviceType>
void
FindKNearestKdTreePoints(
		open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
		int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points
) {
	auto dimension_count = (int32_t) kd_tree_points.GetShape(1);

	switch (dimension_count) {
		case 1:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points, [dimension_count] NNRT_DEVICE_WHEN_CUDACC (float* vector_data){
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points, [dimension_count] NNRT_DEVICE_WHEN_CUDACC (float* vector_data){
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points, [dimension_count] NNRT_DEVICE_WHEN_CUDACC (float* vector_data){
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points, [dimension_count] NNRT_DEVICE_WHEN_CUDACC (float* vector_data){
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);

	}


}

} // nnrt::core::kernel::kdtree