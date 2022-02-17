//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/10/22.
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

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>

#include "core/DeviceHeap.h"
#include "core/kernel/KnnUtilities.h"


namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt::core::kernel::knn {

typedef DistanceIndexPair<float, int32_t> FloatDistanceIndexPair;
typedef decltype(core::MaxHeapKeyCompare<DistanceIndexPair < float, int32_t>>)
NeighborCompare;
template<open3d::core::Device::DeviceType TDeviceType>
using NearestNeighborHeap = core::DeviceHeap<TDeviceType, FloatDistanceIndexPair, NeighborCompare>;

template<o3c::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void ProcessPointEuclideanKnn_BruteForce_PriorityQueue(NearestNeighborHeap<TDeviceType>& nearest_neighbor_heap, int32_t* nearest_neighbor_indices,
                                                              float* squared_distances, const int k, TPointVector& query_point,
                                                              const open3d::t::geometry::kernel::NDArrayIndexer& reference_point_indexer,
                                                              const int i_reference_point, TMakePointVector&& make_point_vector) {
	auto reference_point_data = reference_point_indexer.GetDataPtr<float>(i_reference_point);
	auto reference_point = make_point_vector(reference_point_data);
	float squared_distance = (reference_point - query_point).squaredNorm();
	float max_squared_distance = nearest_neighbor_heap.Head().distance;
	if (squared_distance < max_squared_distance) {
		nearest_neighbor_heap.Pop();
		nearest_neighbor_heap.Insert({squared_distance, i_reference_point});
	}
}

template<o3c::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_BruteForce_PriorityQueue(int32_t* nearest_neighbor_indices, float* squared_distances, const int k,
                                                      const int reference_point_count, TPointVector& query_point,
                                                      const NDArrayIndexer& reference_point_indexer,
                                                      TMakePointVector&& make_point_vector) {
	SetArrayElementsToValue<TDeviceType>(squared_distances, k, INFINITY);

	auto* nearest_neighbor_data = new DistanceIndexPair<float, int32_t>[k];

	core::DeviceHeap<TDeviceType, DistanceIndexPair<float, int32_t>,
			decltype(core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>)>
			nearest_neighbor_heap(nearest_neighbor_data, k, core::MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>);

	for (int i_value = 0; i_value < k; i_value++) {
		nearest_neighbor_heap.Insert(FloatDistanceIndexPair{INFINITY, -1});
	}

	for (int i_reference_point = 0; i_reference_point < reference_point_count; i_reference_point++) {
		ProcessPointEuclideanKnn_BruteForce_PriorityQueue<TDeviceType, TPointVector>(
				nearest_neighbor_heap, nearest_neighbor_indices, squared_distances, k,
				query_point, reference_point_indexer, i_reference_point, make_point_vector);
	}

	const int neighbor_count = nearest_neighbor_heap.Size();
	int i_neighbor = neighbor_count - 1;

	while (!nearest_neighbor_heap.Empty()) {
		FloatDistanceIndexPair pair = nearest_neighbor_heap.Pop();
		nearest_neighbor_indices[i_neighbor] = pair.value;
		squared_distances[i_neighbor] = pair.key;
		i_neighbor--;
	}

	delete[] nearest_neighbor_data;
}

/**
 * \brief find K nearest neighbors by shortest path distance within the provided graph.
 * Edges in the graph are expected to be in a N x GRAPH_DEGREE array, where each row contains indices of target nodes that share an edge with the
 * node at the row index.
 * \tparam TDeviceType
 * \param nearest_neighbor_indices
 * \param distances
 * \param k
 * \param node_count
 * \param point
 * \param reference_point_indexer
 * \param edge_indexer
 */
template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindShortestPathKnn_PriorityQueue(int32_t* nearest_neighbor_indices, float* distances, const int k,
                                              const int node_count, const Eigen::Vector3f& point,
                                              const NDArrayIndexer& reference_point_indexer,
                                              const NDArrayIndexer& edge_indexer, const int graph_degree) {
	int discovered_k = 0;
	typedef core::DistanceIndexPair<float, int32_t> DistanceIndexPair;
	typedef decltype(core::MinHeapKeyCompare<DistanceIndexPair>) Compare;

	const int queue_capacity = 40;
	DistanceIndexPair queue_data[queue_capacity];
	core::DeviceHeap<TDeviceType, DistanceIndexPair, Compare> priority_queue(queue_data, queue_capacity, core::MinHeapKeyCompare<DistanceIndexPair>);

	while (discovered_k < k) {
		int closest_neighbor_index = -1;
		float closest_node_distance;
		FindEuclideanKnn_BruteForce_ExcludingSet<TDeviceType>(
				&closest_neighbor_index, &closest_node_distance, 1, node_count,
				point, reference_point_indexer, nearest_neighbor_indices, k
		);
		closest_node_distance = sqrtf(closest_node_distance);
		if (closest_neighbor_index == -1) {
			break; // no node to initialize queue with, we've got no more valid neighbors to consider
		}
		priority_queue.Insert(DistanceIndexPair{closest_node_distance, closest_neighbor_index});

		while (!priority_queue.Empty() && discovered_k < k) {
			auto source_pair = priority_queue.Pop();
			bool node_already_processed = false;
			for (int i_neighbor = 0; i_neighbor < discovered_k && !node_already_processed; i_neighbor++) {
				if (nearest_neighbor_indices[i_neighbor] == source_pair.value) {
					node_already_processed = true;
				}
			}
			if (node_already_processed) {
				continue;
			}
			distances[discovered_k] = source_pair.key;
			nearest_neighbor_indices[discovered_k] = source_pair.value;
			discovered_k++;
			if (discovered_k >= k) break;

			auto source_pointer = reference_point_indexer.template GetDataPtr<float>(source_pair.value);
			Eigen::Map<const Eigen::Vector3f> source_node(source_pointer);

			for (int i_edge = 0; i_edge < graph_degree; i_edge++) {
				auto target_index_pointer = edge_indexer.template GetDataPtr<int32_t>(source_pair.value);
				int target_node_index = target_index_pointer[i_edge];
				if (target_node_index > -1) {
					auto target_pointer = reference_point_indexer.template GetDataPtr<float>(target_node_index);
					Eigen::Map<const Eigen::Vector3f> target_node(target_pointer);
					float distance_source_to_target = (target_node - source_node).norm();
					priority_queue.Insert(DistanceIndexPair{source_pair.key + distance_source_to_target, target_node_index});
				}
			}
		}
	}
}

}