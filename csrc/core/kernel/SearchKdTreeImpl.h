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
			FindKnnInKdSubtree < TDeviceType > (node->left_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                    (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
		if (query_point[i_dimension] + max_knn_distance > node_value) {
			FindKnnInKdSubtree < TDeviceType > (node->right_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                    (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
	} else {
		if (query_point[i_dimension] + max_knn_distance < node_value) {
			FindKnnInKdSubtree < TDeviceType > (node->right_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
			                                    (i_dimension + 1) % dimension_count, dimension_count, make_point_vector);
		}
		if (query_point[i_dimension] - max_knn_distance > node_value) {
			FindKnnInKdSubtree < TDeviceType > (node->left_child, nearest_neighbor_heap, query_point, kd_tree_point_indexer,
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


				FindKnnInKdSubtree<TDeviceType>(root_node, nearest_neighbor_heap, query_point, kd_tree_point_indexer, 0, dimension_count,
				                                make_point_vector);
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
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType>(
					closest_indices, squared_distances, query_points, k, index_data, kd_tree_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);

	}


}




} // nnrt::core::kernel::kdtree