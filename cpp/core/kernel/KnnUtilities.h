//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/21.
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

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/kernel/KdTreeUtilities.h"


namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt::core::kernel::knn {

template<o3c::Device::DeviceType TDeviceType, typename TElementType>
NNRT_DEVICE_WHEN_CUDACC
inline void SetArrayElementsToValue(TElementType* array, const int size, const TElementType value) {
	for (int i_value = 0; i_value < size; i_value++) {
		array[i_value] = value;
	}
}

template<o3c::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void ProcessPointEuclideanKnn_BruteForce(int& max_at_index, float& max_squared_distance, int32_t* nearest_neighbor_indices,
                                                float* squared_distances, const int k, TPointVector& query_point,
                                                const open3d::t::geometry::kernel::NDArrayIndexer& reference_point_indexer,
                                                const int i_reference_point, TMakePointVector&& make_point_vector) {
	auto reference_point_data = reference_point_indexer.GetDataPtr<float>(i_reference_point);
	auto reference_point = make_point_vector(reference_point_data);
	float squared_distance = (reference_point - query_point).squaredNorm();
	if (squared_distance < max_squared_distance) {
		squared_distances[max_at_index] = squared_distance;
		nearest_neighbor_indices[max_at_index] = i_reference_point;

		//update the maximum distance within current nearest neighbor collection
		max_at_index = 0;
		max_squared_distance = squared_distances[max_at_index];
		for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
			if (squared_distances[i_neighbor] > max_squared_distance) {
				max_at_index = i_neighbor;
				max_squared_distance = squared_distances[i_neighbor];
			}
		}
	}
}

template<o3c::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_BruteForce(int32_t* nearest_neighbor_indices, float* squared_distances, const int k,
                                        const int reference_point_count, TPointVector& query_point, const NDArrayIndexer& reference_point_indexer,
                                        TMakePointVector&& make_point_vector) {
	SetArrayElementsToValue<TDeviceType>(squared_distances, k, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	for (int i_reference_point = 0; i_reference_point < reference_point_count; i_reference_point++) {
		ProcessPointEuclideanKnn_BruteForce<TDeviceType, TPointVector>(
				max_at_index, max_squared_distance, nearest_neighbor_indices, squared_distances, k,
				query_point, reference_point_indexer, i_reference_point, make_point_vector);
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void ProcessPointEuclideanKnn_BruteForce(int& max_at_index, float& max_squared_distance, int32_t* nearest_neighbor_indices,
                                                float* squared_distances, const int k, const Eigen::Vector3f& query_point,
                                                const open3d::t::geometry::kernel::NDArrayIndexer& reference_point_indexer, const int i_point) {
	auto point_data = reference_point_indexer.GetDataPtr<float>(i_point);
	Eigen::Map<Eigen::Vector3f> point(point_data);
	float squared_distance = (point - query_point).squaredNorm();
	if (squared_distance < max_squared_distance) {
		squared_distances[max_at_index] = squared_distance;
		nearest_neighbor_indices[max_at_index] = i_point;

		//update the maximum distance within current nearest neighbor set
		max_at_index = 0;
		max_squared_distance = squared_distances[max_at_index];
		for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
			if (squared_distances[i_neighbor] > max_squared_distance) {
				max_at_index = i_neighbor;
				max_squared_distance = squared_distances[i_neighbor];
			}
		}
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_BruteForce(int32_t* nearest_neighbor_indices, float* squared_distances, const int k,
                                        const int reference_point_count, const Eigen::Vector3f& query_point,
                                        const NDArrayIndexer& reference_point_indexer) {
	SetArrayElementsToValue<TDeviceType>(squared_distances, k, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	for (int i_reference_point = 0; i_reference_point < reference_point_count; i_reference_point++) {
		ProcessPointEuclideanKnn_BruteForce<TDeviceType>(max_at_index, max_squared_distance, nearest_neighbor_indices, squared_distances, k,
		                                                 query_point, reference_point_indexer, i_reference_point);
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_BruteForce_ExcludingSet(int32_t* nearest_neighbor_indices, float* squared_distances, const int k,
                                                     const int reference_point_count, const Eigen::Vector3f& query_point,
                                                     const NDArrayIndexer& reference_point_indexer, const int32_t* excluded_set,
                                                     const int excluded_set_size) {
	SetArrayElementsToValue<TDeviceType>(squared_distances, k, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	for (int i_reference_point = 0; i_reference_point < reference_point_count; i_reference_point++) {
		bool excluded = false;
		for (int i_excluded_index = 0; i_excluded_index < excluded_set_size; i_excluded_index++) {
			if (i_reference_point == excluded_set[i_excluded_index]) {
				excluded = true;
			}
		}
		if (excluded) continue;

		ProcessPointEuclideanKnn_BruteForce<TDeviceType>(max_at_index, max_squared_distance, nearest_neighbor_indices, squared_distances, k,
		                                                 query_point, reference_point_indexer, i_reference_point);
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKnn_KdTree(int32_t* nearest_neighbor_indices, float* nearest_neighbor_distances, const kdtree::KdTreeNode* nodes, const int node_count,
                                    const int k, const Eigen::Vector3f& query_point, const NDArrayIndexer& reference_point_indexer) {
	SetArrayElementsToValue<TDeviceType>(nearest_neighbor_distances, k, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	// Allocate traversal stack from thread-local memory,
	// and push nullptr onto the stack to indicate that there are no deferred nodes.
	int32_t stack[NNRT_KDTREE_STACK_SIZE];
	int32_t* stack_cursor = stack;
	*stack_cursor = -1; // push "-1" onto the bottom of the stuck
	stack_cursor++; // advance the stack cursor

	int32_t node_index = 0;
	do {
		const kdtree::KdTreeNode& node = nodes[node_index];
		if (node_index < node_count && !node.Empty()) {
			Eigen::Map<Eigen::Vector3f> node_point(reference_point_indexer.template GetDataPtr<float>(node.point_index));

			float node_distance = (node_point - query_point).norm();
			// update the nearest neighbor heap if the distance is better than the greatest nearest-neighbor distance encountered so far
			if (max_squared_distance > node_distance) {
				nearest_neighbor_distances[max_at_index] = node_distance;
				nearest_neighbor_indices[max_at_index] = node.point_index;

				//update the maximum distance within current nearest neighbor collection
				max_at_index = 0;
				max_squared_distance = nearest_neighbor_distances[max_at_index];
				for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
					if (nearest_neighbor_distances[i_neighbor] > max_squared_distance) {
						max_at_index = i_neighbor;
						max_squared_distance = nearest_neighbor_distances[i_neighbor];
					}
				}
			}

			const uint8_t i_dimension = node.i_split_dimension;
			float split_plane_value = node_point.coeff(i_dimension);
			float query_coordinate = query_point[i_dimension];

			// Query overlaps an internal node => traverse.
			const int32_t left_child_index = kdtree::GetLeftChildIndex(node_index);
			const int32_t right_child_index = kdtree::GetRightChildIndex(node_index);

			bool search_left_first = query_coordinate < split_plane_value;
			bool search_left = false;
			if (query_coordinate - max_squared_distance <= split_plane_value) {
				// circle with max_knn_distance radius around the query point overlaps the left subtree
				search_left = true;
			}
			bool search_right = false;
			if (query_coordinate + max_squared_distance > split_plane_value) {
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


} // namespace nnrt::core::kernel::knn