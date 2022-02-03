//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/2/22.
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
#pragma once

#include <cfloat>

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>

#include "core/kernel/KdTree.h"
#include "core/kernel/KdTreeUtils.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/KeyValuePair.h"
#include "core/DeviceHeap.h"
#include "core/PlatformIndependence.h"
#include "core/kernel/KnnUtilities.h"

//__DEBUG
#ifdef __CUDACC__
#include <open3d/core/CUDAUtils.h>
#endif


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::kdtree {

namespace {

typedef DistanceIndexPair<float, int32_t> KdDistanceIndexPair;
typedef decltype(core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>)
		NeighborCompare;
template<open3d::core::Device::DeviceType TDeviceType>
using NearestNeighborHeap = core::DeviceHeap<TDeviceType, KdDistanceIndexPair, NeighborCompare>;

template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector, typename TGetMaxDistance, typename TUpdateNeighborSet>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_KdTree_Generic(const KdTreeNode* nodes, const int node_count, const TPointVector& query_point,
                                            const o3gk::NDArrayIndexer& kd_tree_point_indexer, TMakePointVector&& make_point_vector,
                                            TGetMaxDistance&& get_max_distance, TUpdateNeighborSet&& update_neighbor_set) {
	// Allocate traversal stack from thread-local memory,
	// and push nullptr onto the stack to indicate that there are no deferred nodes.
	int32_t stack[64];
	int32_t* stack_cursor = stack;
	*stack_cursor = -1; // push "-1" onto the bottom of the stuck
	stack_cursor++; // advance the stack cursor

	int32_t node_index = 0;
	do {
		const KdTreeNode& node = nodes[node_index];
		if (node_index < node_count && !node.Empty()) {
			float max_neighbor_distance = get_max_distance();
			auto node_point = make_point_vector(kd_tree_point_indexer.template GetDataPtr<float>(node.point_index));

			float node_distance = (node_point - query_point).norm();
			// update the nearest neighbor heap if the distance is better than the greatest nearest-neighbor distance encountered so far
			if (max_neighbor_distance > node_distance) {
				update_neighbor_set(node_distance, node.point_index);
				max_neighbor_distance = get_max_distance();
			}

			const uint8_t i_dimension = node.i_split_dimension;
			float split_plane_value = node_point.coeff(i_dimension);
			float query_coordinate = query_point[i_dimension];

			// Query overlaps an internal node => traverse.
			const int32_t left_child_index = GetLeftChildIndex(node_index);
			const int32_t right_child_index = GetRightChildIndex(node_index);

			bool search_left_first = query_coordinate < split_plane_value;
			bool search_left = false;
			if (query_coordinate - max_neighbor_distance <= split_plane_value) {
				// circle with max_knn_distance radius around the query point overlaps the left subtree
				search_left = true;
			}
			bool search_right = false;
			if (query_coordinate + max_neighbor_distance > split_plane_value) {
				// circle with max_knn_distance radius around the query point overlaps the right subtree
				search_right = true;
			}

			if (!search_left && !search_right) {
				// pop from stack: (1) move cursor back to point at previous entry in the stack, (2) dereference
				node_index = *(--stack_cursor);
			} else {
				if (search_left_first) {
					node_index = search_left ? left_child_index : right_child_index;
					if (search_left && search_right) {
						// push right child onto the stack at the current cursor position
						*stack_cursor = right_child_index;
						stack_cursor++; // advance the stack cursor
					}
				} else {
					node_index = search_right ? right_child_index : left_child_index;
					if (search_left && search_right) {
						// push left child onto the stack at the current cursor position
						*stack_cursor = left_child_index;
						stack_cursor++; // advance the stack cursor
					}
				}
			}
		} else {
			node_index = *(--stack_cursor);
		}
	} while (node_index != -1);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_KdTree_Plain(int32_t* nearest_neighbor_indices, float* neighbor_distances, const KdTreeNode* nodes, const int node_count,
                                          const int k, TPointVector& query_point, const o3gk::NDArrayIndexer& reference_point_indexer,
                                          TMakePointVector&& make_point_vector) {
	core::kernel::knn::SetFloatsToValue<TDeviceType>(neighbor_distances, k, INFINITY);
	int max_at_index = 0;
	float max_neighbor_distance = INFINITY;

	FindEuclideanKnn_KdTree_Generic<TDeviceType>(
			nodes, node_count, query_point, reference_point_indexer, make_point_vector,
			[&max_neighbor_distance]() { return max_neighbor_distance; },
			[&max_neighbor_distance, &max_at_index, &nearest_neighbor_indices, &neighbor_distances, &k](float squared_distance, int point_index) {
				neighbor_distances[max_at_index] = squared_distance;
				nearest_neighbor_indices[max_at_index] = point_index;

				//update the maximum distance within current nearest neighbor collection
				max_at_index = 0;
				max_neighbor_distance = neighbor_distances[max_at_index];
				for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
					if (neighbor_distances[i_neighbor] > max_neighbor_distance) {
						max_at_index = i_neighbor;
						max_neighbor_distance = neighbor_distances[i_neighbor];
					}
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindEuclideanKnn_KdTree_PriorityQueue(int32_t* nearest_neighbor_indices, float* neighbor_distances, const KdTreeNode* nodes, const int node_count,
                                      const int k, TPointVector& query_point, const o3gk::NDArrayIndexer& reference_point_indexer,
                                      TMakePointVector&& make_point_vector) {
	auto* nearest_neighbor_data = new DistanceIndexPair<float, int32_t>[k];

	core::DeviceHeap<TDeviceType, DistanceIndexPair<float, int32_t>,
			decltype(core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>)>
			nearest_neighbor_heap(nearest_neighbor_data, k, core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>);

	for (int i_neighbor = 0; i_neighbor < k; i_neighbor++) {
		nearest_neighbor_heap.Insert(KdDistanceIndexPair{INFINITY, -1});
	}

	FindEuclideanKnn_KdTree_Generic<TDeviceType>(
			nodes, node_count, query_point, reference_point_indexer, make_point_vector,
			[&nearest_neighbor_heap]() { return nearest_neighbor_heap.Head().distance; },
			[&nearest_neighbor_heap](float node_distance, int point_index) {
				nearest_neighbor_heap.Pop();
				nearest_neighbor_heap.Insert(KdDistanceIndexPair{node_distance, point_index});
			}
	);

	const int neighbor_count = nearest_neighbor_heap.Size();
	int i_neighbor = neighbor_count - 1;


	while (!nearest_neighbor_heap.Empty()) {
		KdDistanceIndexPair pair = nearest_neighbor_heap.Pop();
		nearest_neighbor_indices[i_neighbor] = pair.value;
		neighbor_distances[i_neighbor] = pair.key;
		i_neighbor--;
	}
	delete[] nearest_neighbor_data;
}


template<open3d::core::Device::DeviceType TDeviceType, NeighborTrackingStrategy TTrackingStrategy, typename TMakePointVector>
inline void
FindKNearestKdTreePoints_Generic(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& neighbor_distances,
                                 const open3d::core::Tensor& query_points, int32_t k, const KdTreeNode* nodes, const int node_count,
                                 const open3d::core::Tensor& kd_tree_points, TMakePointVector&& make_point_vector) {
	auto query_point_count = query_points.GetLength();


	o3gk::NDArrayIndexer kd_tree_point_indexer(kd_tree_points, 1);
	o3gk::NDArrayIndexer query_point_indexer(query_points, 1);

	nearest_neighbor_indices = open3d::core::Tensor({query_point_count, k}, o3c::Int32, query_points.GetDevice());
	neighbor_distances = open3d::core::Tensor({query_point_count, k}, o3c::Float32, query_points.GetDevice());
	o3gk::NDArrayIndexer nearest_neighbor_indices_indexer(nearest_neighbor_indices, 1);
	o3gk::NDArrayIndexer nearest_neighbor_distances_indexer(neighbor_distances, 1);

	open3d::core::ParallelFor(
			kd_tree_points.GetDevice(), query_point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto query_point = make_point_vector(query_point_indexer.template GetDataPtr<float>(workload_idx));
				auto* indices_for_query_point = nearest_neighbor_indices_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto* squared_distances_for_query_point = nearest_neighbor_distances_indexer.template GetDataPtr<float>(workload_idx);
				if (TTrackingStrategy == NeighborTrackingStrategy::PRIORITY_QUEUE) {
					FindEuclideanKnn_KdTree_PriorityQueue<TDeviceType>(
							indices_for_query_point, squared_distances_for_query_point, nodes, node_count, k,
							query_point, kd_tree_point_indexer, make_point_vector);
				} else {
					FindEuclideanKnn_KdTree_Plain<TDeviceType>(
							indices_for_query_point, squared_distances_for_query_point, nodes, node_count, k,
							query_point, kd_tree_point_indexer, make_point_vector);
				}
			}
	);
}
} // namespace

template<open3d::core::Device::DeviceType TDeviceType, NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Blob& index_data, const int index_length, open3d::core::Tensor& nearest_neighbor_indices,
                              open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
                              const open3d::core::Tensor& kd_tree_points) {
	auto dimension_count = (int32_t) kd_tree_points.GetShape(1);
	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	switch (dimension_count) {
		case 1:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, nodes, index_length, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, nodes, index_length, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, nodes, index_length, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, nodes, index_length, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);

	}


}


} // nnrt::core::kernel::kdtree