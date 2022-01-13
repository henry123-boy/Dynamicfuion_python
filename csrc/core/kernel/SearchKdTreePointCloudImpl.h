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

#include "core/kernel/KdTreePointCloud.h"
#include "core/kernel/KdTreeUtils.h"
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


template<open3d::core::Device::DeviceType TDeviceType, typename TPoint, typename TQueryPoint>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindKnnInKdSubtreeRecursive(const KdTreePointCloudNode <TPoint>* node, const TQueryPoint& query_point,
                            int& max_at_index, float& max_neighbor_distance, float* nearest_neighbors, float* nearest_neighbor_distances,
                            const int k, int i_dimension, const int dimension_count) {
	if (node == nullptr) {
		return;
	}
	auto node_point = node->point;

	float node_distance = (node_point - query_point).norm();
	if (max_neighbor_distance > node_distance) {
		nearest_neighbor_distances[max_at_index] = node_distance;
		TQueryPoint(nearest_neighbors + max_at_index * dimension_count, dimension_count) = node->point;

		//update the maximum distance within current nearest neighbor collection
		max_at_index = 0;
		max_neighbor_distance = nearest_neighbor_distances[max_at_index];
		for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
			if (nearest_neighbor_distances[i_neighbor] > max_neighbor_distance) {
				max_at_index = i_neighbor;
				max_neighbor_distance = nearest_neighbor_distances[i_neighbor];
			}
		}
	}

	float node_value = node_point.coeff(i_dimension);

	bool search_left_first = false;
	bool search_left = false;
	bool search_right = false;

	if (query_point[i_dimension] < node_value) {
		search_left_first = true;
	}
	if (query_point[i_dimension] - max_neighbor_distance < node_value) {
		search_left = true;
	}
	if (query_point[i_dimension] + max_neighbor_distance > node_value) {
		search_right = true;
	}
	if (search_left_first) {
		if (search_left) {
			FindKnnInKdSubtreeRecursive<TDeviceType>(
					node->left_child, query_point, max_at_index, max_neighbor_distance, nearest_neighbors, nearest_neighbor_distances, k,
					(i_dimension + 1) % dimension_count, dimension_count
			);
		}
		if (search_right) {
			FindKnnInKdSubtreeRecursive<TDeviceType>(
					node->right_child, query_point, max_at_index, max_neighbor_distance, nearest_neighbors, nearest_neighbor_distances, k,
					(i_dimension + 1) % dimension_count, dimension_count
			);
		}
	} else {
		if (search_right) {
			FindKnnInKdSubtreeRecursive<TDeviceType>(
					node->right_child, query_point, max_at_index, max_neighbor_distance, nearest_neighbors, nearest_neighbor_distances, k,
					(i_dimension + 1) % dimension_count, dimension_count
			);
		}
		if (search_left) {
			FindKnnInKdSubtreeRecursive<TDeviceType>(
					node->left_child, query_point, max_at_index, max_neighbor_distance, nearest_neighbors, nearest_neighbor_distances, k,
					(i_dimension + 1) % dimension_count, dimension_count
			);
		}
	}
}

template<open3d::core::Device::DeviceType TDeviceType, typename TPoint, typename TQueryPoint>
NNRT_DEVICE_WHEN_CUDACC
inline void FindKnnInKdSubtreeIterative(const KdTreePointCloudNode <TPoint>* root, const TQueryPoint& query_point, int& max_at_index,
                                        float& max_node_distance, float* nearest_neighbors, float* nearest_neighbor_distances,
                                        const int k, const int dimension_count) {

	// Allocate traversal stack from thread-local memory,
	// and push nullptr onto the stack to indicate that there are no deferred nodes.
	const KdTreePointCloudNode<TPoint>* stack[64];
	const KdTreePointCloudNode<TPoint>** stack_cursor = stack;
	*stack_cursor = nullptr; // push nullptr onto the bottom of the stuck
	stack_cursor++; // advance the stack cursor

	const KdTreePointCloudNode<TPoint>* node = root;
	do {
		auto node_point = node->point;

		float node_distance = (node_point - query_point).norm();
		// update the nearest neighbor heap if the distance is better than the greatest nearest-neighbor distance encountered so far
		if (max_node_distance > node_distance) {
			nearest_neighbor_distances[max_at_index] = node_distance;
			TQueryPoint(nearest_neighbors + max_at_index * dimension_count, dimension_count) = node->point;

			//update the maximum distance within current nearest neighbor collection
			max_at_index = 0;
			max_node_distance = nearest_neighbor_distances[max_at_index];
			for (int i_neighbor = 1; i_neighbor < k; i_neighbor++) {
				if (nearest_neighbor_distances[i_neighbor] > max_node_distance) {
					max_at_index = i_neighbor;
					max_node_distance = nearest_neighbor_distances[i_neighbor];
				}
			}
		}

		const uint8_t i_dimension = node->i_dimension;
		float node_coordinate = node_point.coeff(i_dimension);

		// Query overlaps an internal node => traverse.
		float query_coordinate = query_point.coeff(i_dimension);
		const KdTreePointCloudNode<TPoint>* left_child = node->left_child;
		const KdTreePointCloudNode<TPoint>* right_child = node->right_child;

		bool search_left_first = query_coordinate < node_coordinate;
		bool search_left = false;
		if (query_coordinate - max_node_distance <= node_coordinate && left_child != nullptr) {
			// circle with max_knn_distance radius around the query point overlaps the left subtree
			search_left = true;
		}
		bool search_right = false;
		if (query_coordinate + max_node_distance > node_coordinate && right_child != nullptr) {
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

template<open3d::core::Device::DeviceType TDeviceType, SearchStrategy TSearchStrategy, typename TPoint, typename TQueryPoint>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindEuclideanKnn_KdTreePointCloud(float* nearest_neighbors, float* nearest_neighbor_distances, const KdTreePointCloudNode <TPoint>* root_node,
                                  const int k, TQueryPoint& query_point, const int dimension_count) {
	core::kernel::knn::SetFloatsToValue<TDeviceType>(nearest_neighbor_distances, k, INFINITY);
	int max_at_index = 0;
	float max_node_distance = INFINITY;
	if (TSearchStrategy == SearchStrategy::RECURSIVE) {
		FindKnnInKdSubtreeRecursive<TDeviceType>(root_node, query_point, max_at_index, max_node_distance, nearest_neighbors,
		                                         nearest_neighbor_distances, k, 0, dimension_count);
	} else {
		FindKnnInKdSubtreeIterative<TDeviceType>(root_node, query_point, max_at_index, max_node_distance, nearest_neighbors,
		                                         nearest_neighbor_distances, k, dimension_count);
	}
}


template<open3d::core::Device::DeviceType TDeviceType, SearchStrategy TSearchStrategy, typename TPoint, typename TMakeQueryPoint>
inline void
FindKNearestKdTreePoints_Generic(
		open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points,
		int32_t k, int dimension_count, const KdTreePointCloudNode<TPoint>* root_node, TMakeQueryPoint&& make_query_point
) {
	auto query_point_count = query_points.GetLength();
	o3gk::NDArrayIndexer query_point_indexer(query_points, 1);

	nearest_neighbors = open3d::core::Tensor({query_point_count, k, dimension_count}, o3c::Float32, query_points.GetDevice());
	nearest_neighbor_distances = open3d::core::Tensor({query_point_count, k}, o3c::Float32, query_points.GetDevice());
	o3gk::NDArrayIndexer neighbor_indexer(nearest_neighbors, 1);
	o3gk::NDArrayIndexer neighbor_distance_indexer(nearest_neighbor_distances, 1);

#if defined(__CUDACC__)
	namespace launcher = o3c::kernel::cuda_launcher;
#else
	namespace launcher = o3c::kernel::cpu_launcher;
#endif

	launcher::ParallelFor(
			query_point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto query_point = make_query_point(query_point_indexer.template GetDataPtr<float>(workload_idx));
				auto* nearest_neighbors_for_query_point = neighbor_indexer.template GetDataPtr<float>(workload_idx);
				auto* nearest_neighbor_distances_for_query_point = neighbor_distance_indexer.template GetDataPtr<float>(workload_idx);
				FindEuclideanKnn_KdTreePointCloud<TDeviceType, TSearchStrategy>(
						nearest_neighbors_for_query_point, nearest_neighbor_distances_for_query_point, root_node, k,
						query_point, dimension_count);
			}
	);
}
} // namespace

template<open3d::core::Device::DeviceType TDeviceType, SearchStrategy TSearchStrategy>
void
FindKNearestKdTreePointCloudPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
                                   const open3d::core::Tensor& query_points, int32_t k, const void* root, int dimension_count) {

	switch (dimension_count) {
		case 1:
			FindKNearestKdTreePoints_Generic<TDeviceType, TSearchStrategy>(
					nearest_neighbors, nearest_neighbor_distances, query_points, k, dimension_count,
					reinterpret_cast<const KdTreePointCloudNode<Eigen::Vector<float, 1>>*>(root),
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType, TSearchStrategy>(
					nearest_neighbors, nearest_neighbor_distances, query_points, k, dimension_count,
					reinterpret_cast<const KdTreePointCloudNode<Eigen::Vector2f>*>(root),
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType, TSearchStrategy>(
					nearest_neighbors, nearest_neighbor_distances, query_points, k, dimension_count,
					reinterpret_cast<const KdTreePointCloudNode<Eigen::Vector3f>*>(root),
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType, TSearchStrategy>(
					nearest_neighbors, nearest_neighbor_distances, query_points, k, dimension_count,
					reinterpret_cast<const KdTreePointCloudNode<Eigen::Vector<float, Eigen::Dynamic>>*>(root),
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);
			break;

	}


}


#ifdef __CUDACC__
template<typename TPoint>
void PointCloudDataToHost_CUDA(open3d::core::Blob& index_data_cpu, const open3d::core::Blob& index_data, int point_count) {
	auto* nodes_cpu = reinterpret_cast<KdTreePointCloudNode<TPoint>*>(index_data_cpu.GetDataPtr());
	auto* nodes = reinterpret_cast<const KdTreePointCloudNode<TPoint>*>(index_data.GetDataPtr());

	namespace launcher = o3c::kernel::cuda_launcher;

	o3c::Tensor child_indices({point_count, 2}, o3c::Dtype::Int32, index_data.GetDevice());
	o3gk::NDArrayIndexer child_index_indexer(child_indices, 1);

	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				const KdTreePointCloudNode<TPoint>& node = nodes[workload_idx];
				auto* child_set = child_index_indexer.GetDataPtr<int32_t>(workload_idx);
				if (node.left_child == nullptr) {
					child_set[0] = -1;
				} else {
					child_set[0] = static_cast<int32_t>(node.left_child - nodes);
				}
				if (node.right_child == nullptr) {
					child_set[1] = -1;
				} else {
					child_set[1] = static_cast<int32_t>(node.right_child - nodes);
				}
			}
	);
	auto host_child_indices = child_indices.To(o3c::Device("CPU:0"));
	o3gk::NDArrayIndexer host_child_index_indexer(host_child_indices, 1);
	for (int i_node = 0; i_node < point_count; i_node++) {
		auto children_offsets = host_child_index_indexer.GetDataPtr<int32_t>(i_node);
		if (children_offsets[0] == -1) {
			nodes_cpu[i_node].left_child = nullptr;
		} else {
			nodes_cpu[i_node].left_child = nodes_cpu + children_offsets[0];
		}
		if (children_offsets[1] == -1) {
			nodes_cpu[i_node].right_child = nullptr;
		} else {
			nodes_cpu[i_node].right_child = nodes_cpu + children_offsets[1];
		}
	}
}

#endif

} // nnrt::core::kernel::kdtree