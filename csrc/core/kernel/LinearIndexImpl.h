//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/8/22.
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

#include "core/kernel/LinearIndex.h"
#include "KnnUtilities.h"
#include "KnnUtilities_PriorityQueue.h"

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::core::kernel::linear_index {

template<open3d::core::Device::DeviceType TDeviceType, NeighborTrackingStrategy TTrackingStrategy, typename TMakePointVector>
inline void FindKNearestKdTreePoints_Generic(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                                             const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& indexed_points,
                                             TMakePointVector&& make_point_vector) {

	auto query_point_count = query_points.GetLength();
	auto point_count = indexed_points.GetLength();

	o3gk::NDArrayIndexer point_indexer(indexed_points, 1);
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
				auto anchor_indices = closest_indices_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto squared_distances = squared_distance_indexer.template GetDataPtr<float>(workload_idx);
				if (TTrackingStrategy == NeighborTrackingStrategy::PLAIN) {
					core::kernel::knn::FindEuclideanKnn_BruteForce<TDeviceType>(
							anchor_indices, squared_distances,
							k, point_count, query_point, point_indexer, make_point_vector);
				} else {
					core::kernel::knn::FindEuclideanKnn_BruteForce_PriorityQueue<TDeviceType>(
							anchor_indices, squared_distances,
							k, point_count, query_point, point_indexer, make_point_vector);
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType, NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                              const open3d::core::Tensor& query_points, int32_t k, const open3d::core::Tensor& indexed_points) {
	auto dimension_count = (int32_t) indexed_points.GetShape(1);
	switch (dimension_count) {
		case 1:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, squared_distances, query_points, k, indexed_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, squared_distances, query_points, k, indexed_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, squared_distances, query_points, k, indexed_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			FindKNearestKdTreePoints_Generic<TDeviceType, TTrackingStrategy>(
					nearest_neighbor_indices, squared_distances, query_points, k, indexed_points,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);
	}
}

} // namespace nnrt::core::kernel::linear_index