//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/3/23.
//  Copyright (c) 2023 Gregory Kramida
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
// stdlib includes
#include <cfloat>

// third-party includes
#include <open3d/core/Device.h>
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>

// local includes
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/Qualifiers.h"

namespace o3c = open3d::core;

//*** === header for kernel usage only. === ***

namespace nnrt::geometry::functional::kernel::sampling::median {

template<open3d::core::Device::DeviceType TDeviceType>
open3d::core::Tensor ComputeBinPointCounts(o3c::Device& device, const o3c::Tensor& bin_indices, int64_t bin_count) {
	auto bin_index_data = bin_indices.GetDataPtr<int32_t>();
	int64_t node_count = bin_indices.GetLength();
	core::AtomicCounterArray<TDeviceType> bin_point_counts(bin_count);
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_node) {
				int32_t i_bin = bin_index_data[i_node];
				bin_point_counts.FetchAdd(i_bin, 1);
			}
	);
	return bin_point_counts.AsTensor(true);
}

/**
 * \brief Sort point indices such that they appear grouped by their specific bin
 */
template<open3d::core::Device::DeviceType TDeviceType>
o3c::Tensor SortPointIndicesByBins(
		o3c::Device& device, const o3c::Tensor& point_bin_indices,
		const o3c::Tensor& bin_start_indices, int64_t bin_count
) {
	int64_t point_count = point_bin_indices.GetLength();
	o3c::Tensor point_indices_sorted({point_count}, o3c::Int32, device);
	auto point_indices_sorted_data = point_indices_sorted.GetDataPtr<int32_t>();
	auto bin_start_index_data = bin_start_indices.GetDataPtr<int32_t>();
	auto point_bin_index_data = point_bin_indices.GetDataPtr<int32_t>();
	core::AtomicCounterArray<TDeviceType> bin_point_counts(bin_count);

	o3c::ParallelFor(
			device, point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_point) {
				int32_t i_bin = point_bin_index_data[i_point];
				int32_t i_point_in_bin = bin_point_counts.FetchAdd(i_bin, 1);
				auto bin_start_index = bin_start_index_data[i_bin];
				point_indices_sorted_data[bin_start_index + i_point_in_bin] = i_point;
			}
	);

	return point_indices_sorted;
}

/**
 * \brief Compute a point-to-point distance matrix for the set of points associated with each bin
 */
template<open3d::core::Device::DeviceType TDeviceType>
o3c::Tensor ComputeBinDistanceMatrices(
		o3c::Device& device,
		const o3c::Tensor& bin_point_counts,
		const o3c::Tensor& bin_start_indices,
		const o3c::Tensor& bin_distance_matrix_end_indices,
		const o3c::Tensor& points,
		const o3c::Tensor& point_indices_bin_sorted,
		int64_t distance_matrix_set_element_count
) {
	o3c::Tensor distance_matrices({distance_matrix_set_element_count}, o3c::Float32, device);

	auto bin_count = static_cast<int32_t>(bin_point_counts.GetLength());
	auto distance_matrix_data = distance_matrices.GetDataPtr<float>();
	auto bin_point_count_data = bin_point_counts.GetDataPtr<int32_t>();
	auto bin_start_index_data = bin_start_indices.GetDataPtr<int32_t>();
	auto bin_distance_matrix_end_data = bin_distance_matrix_end_indices.GetDataPtr<int32_t>();
	auto point_indices_bin_sorted_data = point_indices_bin_sorted.GetDataPtr<int32_t>();
	auto point_data = points.GetDataPtr<float>();


	o3c::ParallelFor(
			device, distance_matrix_set_element_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto i_element = static_cast<int32_t>(workload_idx);
				int32_t i_bin = 0;
				// find bin
				for (; i_bin < bin_count && i_element >= bin_distance_matrix_end_data[i_bin]; i_bin++);
				// find source & target point
				auto bin_start_index = bin_start_index_data[i_bin];
				auto bin_point_count = bin_point_count_data[i_bin];
				int bin_distance_matrix_start_index = i_bin == 0 ? 0 : bin_distance_matrix_end_data[i_bin-1];
				int32_t i_element_in_matrix = i_element - bin_distance_matrix_start_index;
				int32_t i_source_point_in_bin = i_element_in_matrix / bin_point_count;
				int32_t i_target_point_in_bin = i_element_in_matrix % bin_point_count;
				int32_t i_source_point = point_indices_bin_sorted_data[bin_start_index + i_source_point_in_bin];
				int32_t i_target_point = point_indices_bin_sorted_data[bin_start_index + i_target_point_in_bin];
				Eigen::Map<const Eigen::Vector3f> source_point(point_data + i_source_point * 3);
				Eigen::Map<const Eigen::Vector3f> target_point(point_data + i_target_point * 3);
				// compute distance
				distance_matrix_data[workload_idx] = (target_point - source_point).norm();
			}
	);
	return distance_matrices;
}

template<open3d::core::Device::DeviceType TDeviceType>
o3c::Tensor FindMedianPointIndices(
		o3c::Device& device,
		const o3c::Tensor& distance_matrices,
		const o3c::Tensor& bin_point_counts,
		const o3c::Tensor& bin_start_indices,
		const o3c::Tensor& bin_distance_matrix_start_indices,
		const o3c::Tensor& point_bin_indices,
		const o3c::Tensor& point_indices_bin_sorted
) {
	auto bin_count = static_cast<int32_t>(bin_point_counts.GetLength());
	int64_t point_count = point_bin_indices.GetLength();

	auto distance_matrix_data = distance_matrices.GetDataPtr<float>();
	auto bin_point_count_data = bin_point_counts.GetDataPtr<int32_t>();
	auto bin_start_index_data = bin_start_indices.GetDataPtr<int32_t>();
	auto bin_distance_matrix_start_data = bin_distance_matrix_start_indices.GetDataPtr<int32_t>();
	auto point_indices_bin_sorted_data = point_indices_bin_sorted.GetDataPtr<int32_t>();
	auto point_bin_index_data = point_bin_indices.GetDataPtr<int32_t>();

	o3c::Tensor cumulative_distances_from_other_bin_points({point_count}, o3c::Float32, device);
	auto cumulative_distance_data = cumulative_distances_from_other_bin_points.GetDataPtr<float>();

	// sum distances for each source point in each bin distance matrix
	o3c::ParallelFor(
			device, point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int32_t i_point = point_indices_bin_sorted_data[workload_idx];
				int32_t i_bin = point_bin_index_data[i_point];
				int32_t bin_start_index = bin_start_index_data[i_bin];
				int32_t i_point_in_bin = static_cast<int32_t>(workload_idx) - bin_start_index;
				int32_t bin_point_count = bin_point_count_data[i_bin];
				int32_t distance_matrix_row_start_index = bin_distance_matrix_start_data[i_bin] + i_point_in_bin * bin_point_count;
				const float* distance_matrix_row_start = distance_matrix_data + distance_matrix_row_start_index;
				float sum = 0.f;
				for (int i_target_point = 0; i_target_point < bin_point_count; i_target_point++) {
					sum += distance_matrix_row_start[i_target_point];
				}
				cumulative_distance_data[workload_idx] = sum;
			}
	);
	o3c::Tensor bin_median_point_indices({bin_count}, o3c::Int64, device);
	auto bin_median_index_data = bin_median_point_indices.GetDataPtr<int64_t>();
	// find minimum sum
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_bin) {
				int32_t bin_start_index = bin_start_index_data[i_bin];
				int32_t bin_point_count = bin_point_count_data[i_bin];
				float min_distance_sum = FLT_MAX;
				for (int i_point_in_bin = 0; i_point_in_bin < bin_point_count; i_point_in_bin++) {
					int32_t bin_sorted_point_index = bin_start_index + i_point_in_bin;
					float distance_sum = cumulative_distance_data[bin_sorted_point_index];
					if (distance_sum < min_distance_sum) {
						auto i_point = static_cast<int64_t>(point_indices_bin_sorted_data[bin_sorted_point_index]);
						min_distance_sum = distance_sum;
						bin_median_index_data[i_bin] = i_point;
					}
				}
			}
	);
	return bin_median_point_indices;
}


template<open3d::core::Device::DeviceType TDeviceType>
void MedianGridSamplePoints(
		open3d::core::Tensor& sample, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	auto [point_bin_indices, bin_count, point_bin_coord_map] =
			GridBinPoints<TDeviceType, int32_t>(original_points, grid_cell_size, hash_backend, grid_offset);

	o3c::Tensor bin_point_counts = median::ComputeBinPointCounts<TDeviceType>(device, point_bin_indices, bin_count);
	o3c::Tensor bin_start_indices = core::functional::ExclusiveParallelPrefixSum1D(bin_point_counts);

	o3c::Tensor point_indices_bin_sorted = median::SortPointIndicesByBins<TDeviceType>(device, point_bin_indices, bin_start_indices, bin_count);

	o3c::Tensor point_counts_squared = bin_point_counts * bin_point_counts;

	//TODO: optimize: no need to do two separate scans here -- do an exclusive scan on a sub-tensor starting at 1, fill element 0 with 0 manually.
	o3c::Tensor bin_distance_matrix_start_indices = core::functional::ExclusiveParallelPrefixSum1D(point_counts_squared);
	o3c::Tensor bin_distance_matrix_end_indices = core::functional::InclusiveParallelPrefixSum1D(point_counts_squared);


	int64_t distance_matrix_set_element_count = point_counts_squared.Sum({0}).ToFlatVector<int32_t>()[0];
	// represent a ragged 3D tensor using a 1D tensor
	o3c::Tensor bin_distance_matrices = median::ComputeBinDistanceMatrices<TDeviceType>(
			device, bin_point_counts, bin_start_indices, bin_distance_matrix_end_indices, original_points,
			point_indices_bin_sorted, distance_matrix_set_element_count
	);

	sample = median::FindMedianPointIndices<TDeviceType>(
			device, bin_distance_matrices, bin_point_counts, bin_start_indices, bin_distance_matrix_start_indices,
			point_bin_indices, point_indices_bin_sorted
	);
}

} // namespace  nnrt::geometry::functional::kernel::sampling::median