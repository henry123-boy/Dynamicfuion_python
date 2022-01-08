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
#include "core/KeyValuePair.h"
#include "core/DeviceHeap.h"
#include "core/PlatformIndependence.h"


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::kdtree {

namespace {

typedef DistanceIndexPair<float, int32_t> KdDistanceIndexPair;
typedef decltype(core::MaxHeapKeyCompare<DistanceIndexPair < float, int32_t>>)
NeighborCompare;
template<open3d::core::Device::DeviceType TDeviceType>
using NearestNeighborHeap = core::DeviceHeap<TDeviceType, KdDistanceIndexPair, NeighborCompare>;

template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
void FindKnnInKdSubtreeRecursive(const KdTreeNode* node, NearestNeighborHeap<TDeviceType> nearest_neighbor_heap,
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

	bool search_left = false;
	bool search_right = false;

	if (query_point[i_dimension] < node_value) {
		if (query_point[i_dimension] - max_knn_distance < node_value) {
			search_left = true;
		}
		if (query_point[i_dimension] + max_knn_distance > node_value) {
			search_right = true;
		}
	} else {
		if (query_point[i_dimension] + max_knn_distance < node_value) {
			search_right = true;
		}
		if (query_point[i_dimension] - max_knn_distance > node_value) {
			search_left = true;
		}
	}
	if (search_left) {
		FindKnnInKdSubtreeRecursive<TDeviceType>(node->left_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
		                                         (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
	}
	if (search_right) {
		FindKnnInKdSubtreeRecursive<TDeviceType>(node->right_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
		                                         (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
	}
}

template<open3d::core::Device::DeviceType TDeviceType, typename TPointVector, typename TMakePointVector>
NNRT_DEVICE_WHEN_CUDACC
void FindKnnInKdSubtreeIterative(const KdTreeNode* root, const KdTreeNode* last_node_to_check,
                                 NearestNeighborHeap<TDeviceType> nearest_neighbor_heap,
                                 const TPointVector& query_point,
                                 const o3gk::NDArrayIndexer& kd_tree_point_indexer,
                                 TMakePointVector&& make_point_vector) {
	// Allocate traversal stack from thread-local memory,
	// and push nullptr onto the stack to indicate that there are no deferred nodes.
	const KdTreeNode* stack[64];
	const KdTreeNode** stack_cursor = stack;
	*stack_cursor = nullptr; // push nullptr onto the bottom of the stuck
	stack_cursor++; // advance the stack cursor

	const KdTreeNode* node = root;
	do {
		float max_knn_distance = nearest_neighbor_heap.Head().distance;
		auto node_point = make_point_vector(kd_tree_point_indexer.template GetDataPtr<float>(node->index));

		if (node <= last_node_to_check) {
			float node_distance = (node_point - query_point).squaredNorm();
			// update the nearest neighbor heap if the distance is better than the greatest nearest-neighbor distance encountered so far
			if (max_knn_distance > node_distance) {
				nearest_neighbor_heap.Pop();
				nearest_neighbor_heap.Insert(KdDistanceIndexPair{node_distance, node->index});
				max_knn_distance = nearest_neighbor_heap.Head().distance;
			}
		}

		const uint8_t i_dimension = node->i_dimension;
		float node_coordinate = node_point.coeff(i_dimension);

		// Query overlaps an internal node => traverse.
		float query_coordinate = query_point.coeff(i_dimension);
		const KdTreeNode* left_child = node->left_child;
		const KdTreeNode* right_child = node->right_child;

		bool search_left_first = query_coordinate < node_coordinate;
		bool search_left = false;
		if (query_coordinate - max_knn_distance <= node_coordinate && left_child != nullptr) {
			// circle with max_knn_distance radius around the query point overlaps the left subtree
			search_left = true;
		}
		bool search_right = false;
		if (query_coordinate + max_knn_distance > node_coordinate && right_child != nullptr) {
			// circle with max_knn_distance radius around the query point overlaps the right subtree
			search_right = true;
		}

		if (!search_left && !search_right) {
			// pop from stack: (1) move cursor back to point at previous entry in the stack, (2) dereference
			node = *(--stack_cursor);
		} else {
			if (search_left_first) {
				node = search_left ? left_child : right_child;
				if (search_left && search_right) {
					// push right child onto the stack at the current cursor position
					*stack_cursor = right_child;
					stack_cursor++; // advance the stack cursor
				}
			} else {
				node = search_right ? right_child : left_child;
				if (search_left && search_right) {
					// push left child onto the stack at the current cursor position
					*stack_cursor = left_child;
					stack_cursor++; // advance the stack cursor
				}
			}
		}
	} while (node != nullptr);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TMakePointVector>
inline void
FindKNearestKdTreePoints_Generic(
		open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
		int32_t k, const open3d::core::Blob& index_data, const KdTreeNode* root_node, const open3d::core::Tensor& kd_tree_points,
		TMakePointVector&& make_point_vector
) {
	auto query_point_count = query_points.GetLength();
	auto kd_tree_point_count = kd_tree_points.GetLength();

	//__DEBUG
	// auto dimension_count = (int32_t) kd_tree_points.GetShape(1);
	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	auto* nodes_end = nodes + kd_tree_point_count;
	o3gk::NDArrayIndexer kd_tree_point_indexer(kd_tree_points, 1);
	o3gk::NDArrayIndexer query_point_indexer(query_points, 1);

	nearest_neighbor_indices = open3d::core::Tensor({query_point_count, k}, o3c::Int32, query_points.GetDevice());
	squared_distances = open3d::core::Tensor({query_point_count, k}, o3c::Float32, query_points.GetDevice());
	o3gk::NDArrayIndexer closest_indices_indexer(nearest_neighbor_indices, 1);
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

				const KdTreeNode* node = nodes_end - 1;
				for (int i_node = 0; i_node < k && node >= root_node; i_node++, node--) {
					auto kd_tree_point = make_point_vector(kd_tree_point_indexer.template GetDataPtr<float>(node->index));
					float distance = (kd_tree_point - query_point).squaredNorm();
					nearest_neighbor_heap.Insert(KdDistanceIndexPair{distance, node->index});
				}
				const KdTreeNode* last_node_to_check = node;

				//__DEBUG
				// FindKnnInKdSubtreeRecursive<TDeviceType>(root_node, nearest_neighbor_heap, query_point, kd_tree_point_indexer, 0, dimension_count,
				//                                          make_point_vector);

				FindKnnInKdSubtreeIterative<TDeviceType>(root_node, last_node_to_check, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
				                                         make_point_vector);

				auto* indices_for_query_point = closest_indices_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto* distances_for_query_point = squared_distance_indexer.template GetDataPtr<float>(workload_idx);

				const int neighbor_count = nearest_neighbor_heap.Size();
				int i_neighbor = neighbor_count - 1;



				while (!nearest_neighbor_heap.Empty()) {
					KdDistanceIndexPair pair = nearest_neighbor_heap.Pop();
					indices_for_query_point[i_neighbor] = pair.value;
					distances_for_query_point[i_neighbor] = pair.key;
					i_neighbor--;
				}

				delete[] nearest_neighbor_data;
				for (i_neighbor = neighbor_count; i_neighbor < k; i_neighbor++) {
					indices_for_query_point[i_neighbor] = -1;
					distances_for_query_point[i_neighbor] = FLT_MAX;
				}
			}
	);
}
} // namespace

template<open3d::core::Device::DeviceType TDeviceType>
void
FindKNearestKdTreePoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points,
                         int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points, const void* root) {
	auto dimension_count = (int32_t) kd_tree_points.GetShape(1);
	auto* root_node = reinterpret_cast<const KdTreeNode*>(root);
	switch (dimension_count) {
		case 1:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					nearest_neighbor_indices, squared_distances, query_points, k, index_data, root_node, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					nearest_neighbor_indices, squared_distances, query_points, k, index_data, root_node, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					nearest_neighbor_indices, squared_distances, query_points, k, index_data, root_node, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					nearest_neighbor_indices, squared_distances, query_points, k, index_data, root_node, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);

	}


}


} // nnrt::core::kernel::kdtree