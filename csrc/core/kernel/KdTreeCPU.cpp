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

#include <atomic>
#include <open3d/core/ParallelFor.h>
#include <unordered_set>
#include <mutex>
#include "core/CPU/DeviceHeapCPU.h"
#include "core/kernel/BuildKdTreeImpl.h"
#include "core/kernel/SearchKdTreeImpl.h"
#include "core/kernel/DownsampleKdTreeImpl.h"


namespace nnrt::core::kernel::kdtree {

template
void
BuildKdTreeIndex<open3d::core::Device::DeviceType::CPU>(open3d::core::Blob& index_data, int64_t index_length, const open3d::core::Tensor& points);


template
void FindKNearestKdTreePoints<open3d::core::Device::DeviceType::CPU, NeighborTrackingStrategy::PLAIN>(
		open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
		open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
		const open3d::core::Tensor& reference_points);

template
void FindKNearestKdTreePoints<open3d::core::Device::DeviceType::CPU, NeighborTrackingStrategy::PRIORITY_QUEUE>(
		open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
		open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
		const open3d::core::Tensor& reference_points);

template
void DecimateReferencePoints<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& decimated_points, open3d::core::Blob& index_data, int node_count,
		const open3d::core::Tensor& reference_points, float downsampling_radius);

namespace {
template<typename TPoint>
void ProcessRadiusNeighborsForDecimation_CPU_Generic(
		open3d::core::Tensor& decimated_points, const open3d::core::Tensor& reference_points,
		const open3d::core::Tensor& radius_neighbors, float downsampling_radius) {
	const int64_t reference_point_count = reference_points.GetShape(0);
	auto processed_points = open3d::core::Tensor({reference_point_count}, o3c::Dtype::Int32, reference_points.GetDevice());
	int processed_point_count = 0;
	std::mutex processed_point_mutex;

	std::unordered_set<int32_t> eliminated_point_set;

	auto point_mask = o3c::Tensor({reference_point_count}, o3c::Dtype::Bool, reference_points.GetDevice());
	point_mask.template Fill(false);

	NDArrayIndexer processed_point_indexer(processed_points, 1);
	NDArrayIndexer radius_neighbor_indexer(radius_neighbors, 1);
	NDArrayIndexer reference_point_indexer(reference_points, 1);
	NDArrayIndexer point_mask_indexer(point_mask, 1);


	open3d::core::ParallelFor(
			reference_points.GetDevice(), reference_point_count,
			[&](int64_t workload_idx) {
				// skip over the point if we'd already eliminated it.
				if (eliminated_point_set.find((int32_t) workload_idx) != eliminated_point_set.end()) {
					return;
				}
				auto query_point_data = reference_point_indexer.template GetDataPtr<float>(workload_idx);
				auto query_point = Eigen::Map<Eigen::Vector3f>(query_point_data);

				// check if the current point is within range of those being processed.
				// if not in range and processed points array hasn't been modified, add it to the processed points
				// array within a synchronized block and proceed.
				bool in_range_of_processed = false;
				int already_checked_count = 0;
				int current_count;
				do {
					current_count = processed_point_count;
					for (; already_checked_count < current_count && !in_range_of_processed; already_checked_count++) {
						auto processed_point = Eigen::Map<Eigen::Vector3f>(
								reference_point_indexer.template GetDataPtr<float>(
										*processed_point_indexer.template GetDataPtr<int32_t>(already_checked_count)
								)
						);
						if ((processed_point - query_point).norm() < downsampling_radius) {
							in_range_of_processed = true;
						}
					}
				} while (!in_range_of_processed && ![&]() {
					std::lock_guard<std::mutex> lock(processed_point_mutex);
					if (current_count != processed_point_count) {
						return false;
					};
					auto processed_point_data = processed_point_indexer.template GetDataPtr<int>(current_count);
					*processed_point_data = static_cast<int32_t>(workload_idx);
					processed_point_count++;
					return true;
				}());

				auto* point_radius_neighbors = radius_neighbor_indexer.GetDataPtr<int>(workload_idx);
				int i_neighbor = 0;
				while (i_neighbor < NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS && point_radius_neighbors[i_neighbor] != -1) {
					eliminated_point_set.insert(point_radius_neighbors[i_neighbor]);
				}
				*point_mask_indexer.template GetDataPtr<bool>(workload_idx) = true;
			}

	);

	decimated_points = reference_points.GetItem(o3c::TensorKey::IndexTensor(point_mask));
}

} // namespace

template<>
void ProcessRadiusNeighborsForDecimation<open3d::core::Device::DeviceType::CPU, Eigen::Vector3f>(
		open3d::core::Tensor& decimated_points, const open3d::core::Tensor& reference_points,
		const open3d::core::Tensor& radius_neighbors, float downsampling_radius) {
}


} // nnrt::core::kernel::kdtree