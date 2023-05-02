//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/25/22.
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
//TODO: remove header / replace implementation if unused
//stdlib
#include <cmath>
#include <cfloat>

// 3rd party
#include <open3d/core/hashmap/HashMap.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

// local
#include "GeometrySampling.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/platform_independence/Atomics.h"
#include "core/kernel/HashTableUtilities.h"
#include "geometry/functional/kernel/PointAggregationBins.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/functional/ExclusiveParallelPrefixScan.h"

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;
namespace hash_table = nnrt::core::kernel::hash_table;
namespace utility = open3d::utility;

namespace nnrt::geometry::functional::kernel::sampling {

namespace {

template<open3d::core::Device::DeviceType DeviceType>
void InitializeAveragingBins(
		const o3c::Device& device,
		int64_t bin_count,
		PointAverageAggregationBin* bins
) {
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAverageAggregationBin& bin = bins[workload_idx];
				bin.x = 0.f;
				bin.y = 0.f;
				bin.z = 0.f;
				bin.count = 0;
			}
	);
}

template<open3d::core::Device::DeviceType DeviceType>
void ComputeBinAverageAggregates(
		const o3gk::NDArrayIndexer& original_point_indexer,
		const o3c::Device& device,
		int64_t original_point_count,
		PointAverageAggregationBin* bins,
		int32_t* bin_indices
) {
	o3c::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
				auto bin_index = bin_indices[workload_idx];
				PointAverageAggregationBin& bin = bins[bin_index];
#ifdef __CUDACC__
				atomicAdd(&bin.x, point.x());
				atomicAdd(&bin.y, point.y());
				atomicAdd(&bin.z, point.z());
				atomicAdd(&bin.count, 1);
#else
				atomicAdd_CPU(bin.x, point.x());
				atomicAdd_CPU(bin.y, point.y());
				atomicAdd_CPU(bin.z, point.z());
				bin.count++;
#endif
			}
	);
}


template<o3c::Device::DeviceType DeviceType>
void ComputeBinAverages(
		o3c::Device& device,
		int64_t bin_count,
		PointAverageAggregationBin* bins
) {

	// COMPUTE BIN AVERAGES
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAverageAggregationBin& bin = bins[workload_idx];
#ifdef __CUDACC__
				bin.x /= static_cast<float>(bin.count);
				bin.y /= static_cast<float>(bin.count);
				bin.z /= static_cast<float>(bin.count);
#else
				int bin_point_count = bin.count.load();
				bin.x.store(bin.x.load() / static_cast<float>(bin_point_count));
				bin.y.store(bin.y.load() / static_cast<float>(bin_point_count));
				bin.z.store(bin.z.load() / static_cast<float>(bin_point_count));
#endif
			}
	);
}


template<o3c::Device::DeviceType DeviceType>
void TransferFromBinsToTensor_NoCountCheck(
		o3gk::NDArrayIndexer& downsampled_point_indexer,
		o3c::Device& device,
		int64_t bin_count,
		PointAverageAggregationBin* bins
) {

	// STORE POINTS IN RESULT TENSOR
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAverageAggregationBin& bin = bins[workload_idx];
				float x, y, z;
#ifdef __CUDACC__
				x = bin.x; y = bin.y; z = bin.z;
#else
				x = bin.x.load();
				y = bin.y.load();
				z = bin.z.load();
#endif
				Eigen::Map<Eigen::Vector3f> downsampled_point(downsampled_point_indexer.GetDataPtr<float>(workload_idx));
				downsampled_point.x() = x;
				downsampled_point.y() = y;
				downsampled_point.z() = z;
			}
	);
}

template<o3c::Device::DeviceType DeviceType>
void TransferFromBinsToTensor_CountCheck(
		o3gk::NDArrayIndexer& downsampled_point_indexer,
		o3c::Device& device,
		int64_t bin_count,
		PointAverageAggregationBin* bins
) {

	NNRT_DECLARE_ATOMIC(uint32_t, downsampled_point_count_atomic);
	NNRT_INITIALIZE_ATOMIC(unsigned int, downsampled_point_count_atomic, 0);
	// STORE POINTS IN RESULT TENSOR
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAverageAggregationBin& bin = bins[workload_idx];
				int64_t downsampled_point_index;
				bool has_point = false;
				float x, y, z;
#ifdef __CUDACC__
				if(bin.count != 0){
					has_point = true;
					x = bin.x; y = bin.y; z = bin.z;
				}
#else
				if (bin.count.load() != 0) {
					has_point = true;
					x = bin.x.load();
					y = bin.y.load();
					z = bin.z.load();
				}
#endif
				if (!has_point)
					return;

				downsampled_point_index = NNRT_ATOMIC_ADD(downsampled_point_count_atomic, (uint32_t) 1);
				Eigen::Map<Eigen::Vector3f> downsampled_point(downsampled_point_indexer.GetDataPtr<float>(downsampled_point_index));
				downsampled_point.x() = x;
				downsampled_point.y() = y;
				downsampled_point.z() = z;

			}
	);NNRT_CLEAN_UP_ATOMIC(downsampled_point_count_atomic);
}

template<open3d::core::Device::DeviceType TDeviceType, typename TBinType>
std::tuple<open3d::core::Tensor, int64_t, o3c::HashMap> GridBinPoints(
		const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	o3c::Tensor grid_offset_tensor(std::vector<float>{grid_offset.x(), grid_offset.y(), grid_offset.z()}, {1, 3}, o3c::Float32, device);

	o3c::Tensor point_bins_float = original_points / grid_cell_size + grid_offset_tensor;
	o3c::Tensor point_bins_integer = point_bins_float.Floor().To(o3c::Int32);

	o3c::HashMap point_bin_coord_map(point_bins_integer.GetLength(), o3c::Int32, {3}, o3c::UInt8, {sizeof(TBinType)}, device,
	                                 hash_backend);

	o3c::Tensor bin_indices_tensor, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, bin_indices_tensor, success_mask);

	o3c::Tensor bins_integer = point_bins_integer.GetItem(o3c::TensorKey::IndexTensor(success_mask));
	auto bin_count = bins_integer.GetLength();

	std::tie(bin_indices_tensor, success_mask) = point_bin_coord_map.Find(point_bins_integer);
	return std::make_tuple(bin_indices_tensor, bin_count, point_bin_coord_map);
}


template<open3d::core::Device::DeviceType TDeviceType>
void AveragePointsIntoGridCells(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	auto [bin_indices_tensor, bin_count, point_bin_coord_map] = GridBinPoints<TDeviceType, PointAverageAggregationBin>(original_points,
	                                                                                                                   grid_cell_size, hash_backend,
	                                                                                                                   grid_offset);

	auto bin_indices = bin_indices_tensor.template GetDataPtr<int32_t>();
	auto* bins = reinterpret_cast<PointAverageAggregationBin*>(point_bin_coord_map.GetValueTensor().GetDataPtr());
	InitializeAveragingBins<TDeviceType>(device, bin_count, bins);

	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();

	ComputeBinAverageAggregates<TDeviceType>(original_point_indexer, device, original_point_count, bins, bin_indices);
	ComputeBinAverages<TDeviceType>(device, bin_count, bins);

	downsampled_points = o3c::Tensor({bin_count, 3}, o3c::Float32, device);
	o3gk::NDArrayIndexer downsampled_point_indexer(downsampled_points, 1);

	TransferFromBinsToTensor_NoCountCheck<TDeviceType>(downsampled_point_indexer, device, bin_count, bins);
}
//
// template<open3d::core::Device::DeviceType TDeviceType>
// void FindGridCellRadiusClusters(
// 		const open3d::core::Tensor& vertices, const open3d::core::Tensor& edges, float grid_cell_size, float radius,
// 		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
// 	auto device = vertices.GetDevice();
//
// 	auto [bin_indices_tensor, bin_count, point_bin_coord_map] =
// 			GridBinPoints<TDeviceType, PointCollectionBin>(vertices, grid_cell_size, hash_backend, grid_offset);
//
// 	auto* bins = reinterpret_cast<PointCollectionBin*>(point_bin_coord_map.GetValueTensor().GetDataPtr());
//
// 	auto bin_index_data = bin_indices_tensor.template GetDataPtr<int32_t>();
//
// 	int64_t vertex_count = vertices.GetLength();
// 	o3c::ParallelFor(
// 			device, vertex_count,
// 			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_vertex) {
// 				int32_t i_bin = bin_index_data[i_vertex];
// 				bins[i_bin].UpdateWithPoint<TDeviceType>(i_vertex);
// 			}
// 	);
//
//
// 	int64_t max_vertex_degree = edges.GetShape(1);
// 	auto edge_data = edges.GetDataPtr<int64_t>();
// 	o3c::Tensor vertex_cluster_assignments({vertex_count}, o3c::Int32, device);
// 	vertex_cluster_assignments.Fill(-1);
// 	auto vertex_cluster_data = vertex_cluster_assignments.GetDataPtr<int32_t>();
//
// 	auto vertex_data = vertices.GetDataPtr<float>();
//
// 	NNRT_DECLARE_ATOMIC_INT(cluster_count);
//
// 	o3c::ParallelFor(
// 			device, bin_count,
// 			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_bin) {
// 				PointCollectionBin& bin = bins[i_bin];
// 				const int bin_point_count = bin.count;
// 				for (int i_source_in_bin = 0; i_source_in_bin < bin_point_count; i_source_in_bin++) {
// 					int64_t source_vertex_index = bin.indices[i_source_in_bin];
// 					const int64_t* vertex_edges = edge_data + source_vertex_index * max_vertex_degree;
// 					int cluster_source = vertex_cluster_data[source_vertex_index];
// 					Eigen::Map<const Eigen::Vector3f> source_vertex(vertex_data + source_vertex_index * 3);
// 					for (int i_target = 0; i_target < max_vertex_degree; i_target++) {
// 						int64_t target_vertex_index = edge_data[i_target];
// 						Eigen::Map<const Eigen::Vector3f> target_vertex(vertex_data + target_vertex_index * 3);
// 						float distance = (source_vertex - target_vertex).norm();
// 						if (distance < radius) {
// 							if (cluster_source == -1) {
// 								int cluster_target = vertex_cluster_data[target_vertex_index];
// 								if (cluster_target == -1) {
// 									int new_cluster_index = NNRT_ATOMIC_ADD(cluster_count, 1);
// 									vertex_cluster_data[source_vertex_index] = cluster_source = new_cluster_index;
// 									vertex_cluster_data[target_vertex_index] = new_cluster_index;
// 								} else {
// 									vertex_cluster_data[source_vertex_index] = cluster_source = cluster_target;
// 								}
// 							} else {
// 								int cluster_target = vertex_cluster_data[target_vertex_index];
// 							}
// 						}
//
// 					}
// 				}
// 			}
// 	);
//
// 	NNRT_CLEAN_UP_ATOMIC(cluster_count);
//
//
// }


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
o3c::Tensor ComputeDistanceMatrices(
		o3c::Device& device,
		const o3c::Tensor& bin_point_counts,
		const o3c::Tensor& bin_start_indices,
		const o3c::Tensor& bin_distance_matrix_start_indices,
		const o3c::Tensor& points,
		const o3c::Tensor& point_indices_bin_sorted,
		int64_t distance_matrix_set_element_count
) {
	o3c::Tensor distance_matrices({distance_matrix_set_element_count}, o3c::Float32, device);

	auto bin_count = static_cast<int32_t>(bin_point_counts.GetLength());
	auto distance_matrix_data = distance_matrices.GetDataPtr<float>();
	auto bin_point_count_data = bin_point_counts.GetDataPtr<int32_t>();
	auto bin_start_index_data = bin_start_indices.GetDataPtr<int32_t>();
	auto bin_distance_matrix_start_data = bin_distance_matrix_start_indices.GetDataPtr<int32_t>();
	auto point_indices_bin_sorted_data = point_indices_bin_sorted.GetDataPtr<int32_t>();
	auto point_data = points.GetDataPtr<float>();


	o3c::ParallelFor(
			device, distance_matrix_set_element_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto i_element = static_cast<int32_t>(workload_idx);
				int32_t i_bin = 0;
				// find bin
				for (; i_bin + 1 < bin_count && i_element > bin_distance_matrix_start_data[i_bin + 1]; i_bin++);
				// find source & target point
				auto bin_start_index = bin_start_index_data[i_bin];
				auto bin_point_count = bin_point_count_data[i_bin];
				auto bin_distance_matrix_start = bin_distance_matrix_start_data[i_bin];
				int32_t i_element_in_matrix = i_element - bin_distance_matrix_start;
				int32_t i_source_point_in_bin = i_element_in_matrix / bin_point_count;
				int32_t i_target_point_in_bin = i_element_in_matrix % bin_point_count;
				int32_t i_source_point = point_indices_bin_sorted_data[bin_start_index + i_source_point_in_bin];
				int32_t i_target_point = point_indices_bin_sorted_data[bin_start_index + i_target_point_in_bin];
				Eigen::Map<const Eigen::Vector3f> source_point(point_data + i_source_point);
				Eigen::Map<const Eigen::Vector3f> target_point(point_data + i_target_point);
				// compute distance
				distance_matrix_data[bin_distance_matrix_start + i_element_in_matrix] = (target_point - source_point).norm();
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
						bin_median_index_data[i_bin] = i_point;
					}
				}
			}
	);
	return bin_distance_matrix_start_indices;
}

} // anonymous namespace


template<open3d::core::Device::DeviceType TDeviceType>
void GridDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend
) {
	AveragePointsIntoGridCells<TDeviceType>(downsampled_points, original_points, grid_cell_size, hash_backend);
}

template<open3d::core::Device::DeviceType TDeviceType>
void FastRadiusDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float radius,
		const open3d::core::HashBackendType& hash_backend
) {
	o3c::Tensor downsampled_points_stage_1;
	float extended_radius = sqrtf(2 * (radius * radius));
	AveragePointsIntoGridCells<TDeviceType>(downsampled_points_stage_1, original_points, extended_radius * 2, hash_backend);
	// merge again while offsetting the grid
	AveragePointsIntoGridCells<TDeviceType>(downsampled_points, downsampled_points_stage_1, extended_radius * 2, hash_backend,
	                                        Eigen::Vector3f(0.5, 0.5, 0.5));
}


template<open3d::core::Device::DeviceType TDeviceType>
void MedianGridSamplePoints(
		open3d::core::Tensor& sample, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	o3c::Tensor grid_offset_tensor(std::vector<float>{grid_offset.x(), grid_offset.y(), grid_offset.z()}, {1, 3}, o3c::Float32, device);

	o3c::Tensor point_bins_float = original_points / grid_cell_size + grid_offset_tensor;
	o3c::Tensor point_bins_integer = point_bins_float.Floor().To(o3c::Int32);

	o3c::HashMap point_bin_coord_map(point_bins_integer.GetLength(), o3c::Int32, {3}, o3c::UInt8, {sizeof(PointAverageAggregationBin)}, device,
	                                 hash_backend);

	// activate entries in the hash map that correspond to the
	o3c::Tensor point_bin_indices, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, point_bin_indices, success_mask);
	o3c::Tensor bins_integer = point_bins_integer.GetItem(o3c::TensorKey::IndexTensor(success_mask));
	auto bin_count = bins_integer.GetLength();

	std::tie(point_bin_indices, success_mask) = point_bin_coord_map.Find(point_bins_integer);
	o3c::Tensor bin_point_counts = ComputeBinPointCounts<TDeviceType>(device, point_bin_indices, bin_count);
	o3c::Tensor bin_start_indices = core::functional::ExclusiveParallelPrefixSum1D(bin_point_counts);

	o3c::Tensor point_indices_bin_sorted = SortPointIndicesByBins<TDeviceType>(device, point_bin_indices, bin_start_indices, bin_count);

	o3c::Tensor point_counts_squared = bin_point_counts * bin_point_counts;
	o3c::Tensor bin_distance_matrix_start_indices = core::functional::ExclusiveParallelPrefixSum1D(point_counts_squared);


	int64_t distance_matrix_set_element_count = point_counts_squared.Sum({0}).ToFlatVector<int32_t>()[0];
	// represent a ragged 3D tensor using a 1D tensor
	o3c::Tensor distance_matrices = ComputeDistanceMatrices<TDeviceType>(
			device, bin_point_counts, bin_start_indices, bin_distance_matrix_start_indices, original_points,
			point_indices_bin_sorted, distance_matrix_set_element_count
	);

	sample = FindMedianPointIndices<TDeviceType>(
			device, distance_matrices, bin_point_counts, bin_start_indices, bin_distance_matrix_start_indices,
			point_bin_indices, point_indices_bin_sorted
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void RadiusMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float radius,
		const open3d::core::HashBackendType& hash_backend_type
) {
	float extended_radius = sqrtf(2 * (radius * radius));
	o3c::Tensor median_point_indices_stage1, median_point_indices_stage2;
	MedianGridSamplePoints<TDeviceType>(median_point_indices_stage1, points, extended_radius * 2, hash_backend_type);
	o3c::Tensor points_stage_1 = points.GetItem(o3c::TensorKey::IndexTensor(median_point_indices_stage1));
	// merge again while offsetting the grid
	MedianGridSamplePoints<TDeviceType>(median_point_indices_stage2, points_stage_1, extended_radius * 2, hash_backend_type,
	                                    Eigen::Vector3f(0.5, 0.5, 0.5));
	sample = median_point_indices_stage1.GetItem(o3c::TensorKey::IndexTensor(median_point_indices_stage2));
}


template<open3d::core::Device::DeviceType TDeviceType>
void RadiusSubsampleGraph(
		open3d::core::Tensor& resampled_vertices,
		open3d::core::Tensor& resampled_edges,
		const open3d::core::Tensor& vertices,
		const open3d::core::Tensor& edges,
		float radius
) {
	// counters and checks
	o3c::Device device = vertices.GetDevice();
	int64_t vertex_count = vertices.GetLength();
	o3c::AssertTensorShape(vertices, { vertex_count, 3 });
	o3c::AssertTensorDtype(vertices, o3c::Float32);

	int64_t max_vertex_degree = edges.GetShape(1);
	o3c::AssertTensorShape(edges, { vertex_count, max_vertex_degree });
	o3c::AssertTensorDtype(edges, o3c::Int64);
	o3c::AssertTensorDevice(edges, device);

	if (radius < 1e-7) {
		utility::LogError("Radius must be a positive value above 1e-7. Provided radius: {}", radius);
	}

	// prep inputs
	auto* vertex_data = vertices.GetDataPtr<float>();
	auto* edge_data = edges.GetDataPtr<int64_t>();

	// init output for this phase

	// find super-independent vertices
	o3c::Tensor super_independent_mask_tier1 = o3c::Tensor::Zeros({vertex_count}, o3c::Bool, device);
	auto* super_independent_mask1_data = super_independent_mask_tier1.GetDataPtr<bool>();
	NNRT_DECLARE_ATOMIC(int, super_independent_count);
	o3c::Tensor super_independent_set = o3c::Tensor({vertex_count}, o3c::Int64, device);
	auto super_independent_set_data = super_independent_set.GetDataPtr<int64_t>();

	// greedy algorithm, prone to race conditions
	o3c::ParallelFor(
			device, vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_source_vertex) {
				if (super_independent_mask1_data[i_source_vertex]) {
					return;
				} else {
					auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
					for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
						int64_t i_target_vertex = vertex_edges[i_edge];
						if (i_target_vertex == -1) {
							break;
						} else if (super_independent_mask1_data[i_target_vertex]) {
							// neighbor already in set, this source should be excluded
							return;
						}
					}
					super_independent_mask1_data[i_source_vertex] = true;
					int index = NNRT_ATOMIC_ADD(super_independent_count, 1);
					super_independent_set_data[index] = i_source_vertex;
				}
			}
	);

	// check for race conditions: remove any extra vertices from the set (which have distance < 3 edges from another vertex in the set)
	o3c::Tensor super_independent_mask_tier2 = o3c::Tensor::Ones({NNRT_GET_ATOMIC_VALUE_HOST(super_independent_count)}, o3c::Bool, device);
	auto* super_independent_mask2_data = super_independent_mask_tier2.GetDataPtr<bool>();
	o3c::ParallelFor(
			device, NNRT_GET_ATOMIC_VALUE_HOST(super_independent_count),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_super_independent_vertex) {
				int64_t i_source_vertex = super_independent_set_data[i_super_independent_vertex];
				auto vertex_edges_1_ring = edge_data + i_source_vertex * max_vertex_degree;
				for (int i_edge_1_ring = 0; i_edge_1_ring < max_vertex_degree; i_edge_1_ring++) {
					int64_t i_target_vertex_1_ring = vertex_edges_1_ring[i_edge_1_ring];
					if (i_target_vertex_1_ring == -1) {
						break;
					} else if (super_independent_mask1_data[i_target_vertex_1_ring] && super_independent_mask2_data[i_target_vertex_1_ring]) {
						// distance is only 1 edge
						super_independent_mask2_data[i_super_independent_vertex] = false;
					} else {
						auto vertex_edges_2_ring = edge_data + i_target_vertex_1_ring * max_vertex_degree;
						for (int i_edge_2_ring = 0; i_edge_2_ring < max_vertex_degree; i_edge_2_ring++) {
							int64_t i_target_vertex_2_ring = vertex_edges_2_ring[i_edge_2_ring];
							if (i_target_vertex_2_ring == -1) {
								break;
							} else if (super_independent_mask1_data[i_target_vertex_2_ring] && super_independent_mask2_data[i_target_vertex_2_ring]) {
								// distance is only 2 edges
								super_independent_mask2_data[i_super_independent_vertex] = false;
							}
						}
					}
				}
			}
	);

	super_independent_set = super_independent_set.GetItem(o3c::TensorKey::IndexTensor(super_independent_mask_tier2));NNRT_CLEAN_UP_ATOMIC(
			super_independent_count);

	int64_t super_independent_vertex_count = super_independent_set.GetLength();
	super_independent_set_data = super_independent_set.GetDataPtr<int64_t>();
	o3c::Tensor new_vertex_indices({vertex_count}, o3c::Int64, device);
	new_vertex_indices.Fill(-1);
	auto new_vertex_index_data = new_vertex_indices.GetDataPtr<int64_t>();
	o3c::Tensor new_vertex_positions({vertex_count, 3}, o3c::Float32, device);
	auto new_vertex_position_data = new_vertex_positions.GetDataPtr<float>();

	NNRT_DECLARE_ATOMIC(int, new_vertex_count);

	// collapse edges around super-independent vertices if they are shorter than radius
	o3c::ParallelFor(
			device, super_independent_vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_super_independent_vertex) {
				int64_t i_source_vertex = super_independent_set_data[i_super_independent_vertex];
				Eigen::Map<const Eigen::Vector3f> source_vertex(vertex_data + i_source_vertex * 3);

				int new_merged_vertex_index = NNRT_ATOMIC_ADD(new_vertex_count, 1);
				new_vertex_index_data[i_source_vertex] = new_merged_vertex_index;

				Eigen::Map<Eigen::Vector3f> averaged_vertex(new_vertex_position_data + new_merged_vertex_index * 3);
				averaged_vertex = Eigen::Vector3f(0.f, 0.f, 0.f);
				averaged_vertex += source_vertex;
				int averaged_vertex_count = 0;

				auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
				for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
					int64_t i_target_vertex = vertex_edges[i_edge];
					Eigen::Map<const Eigen::Vector3f> target_vertex(vertex_data + i_target_vertex * 3);
					if ((source_vertex - target_vertex).norm() < radius) {
						averaged_vertex_count++;
						averaged_vertex += target_vertex;
						new_vertex_index_data[i_target_vertex] = new_merged_vertex_index;
					} else {
						int unmerged_vertex_index = NNRT_ATOMIC_ADD(new_vertex_count, 1);
						new_vertex_index_data[i_target_vertex] = unmerged_vertex_index;
						memcpy(new_vertex_position_data + unmerged_vertex_index * 3, vertex_data + i_target_vertex * 3, 3 * sizeof(float));
					}
				}
				averaged_vertex /= static_cast<float>(averaged_vertex_count);
			}
	);

	int64_t new_max_vertex_degree = max_vertex_degree * 4;

	resampled_vertices = new_vertex_positions.Slice(0, 0, NNRT_GET_ATOMIC_VALUE_HOST(new_vertex_count));NNRT_CLEAN_UP_ATOMIC(new_vertex_count);

	resampled_edges = o3c::Tensor({vertices.GetLength(), new_max_vertex_degree}, o3c::Int64, device);
	resampled_edges.Fill(-1);
	auto new_edge_data = resampled_edges.GetDataPtr<int64_t>();
	core::AtomicCounterArray<TDeviceType> new_edge_counts(vertices.GetLength());

	// reconstruct edges from new indices
	o3c::ParallelFor(
			device, vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_source_vertex) {
				int64_t i_new_source_vertex = new_vertex_index_data[i_source_vertex];
				auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
				auto new_vertex_edges = new_edge_data + i_new_source_vertex * new_max_vertex_degree;
				for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
					int64_t i_target_vertex = vertex_edges[i_edge];
					int64_t i_new_target_vertex = new_vertex_index_data[i_target_vertex];
					bool already_added = false;
					for (int i_new_edge = 0; i_new_edge < new_edge_counts.GetCount(i_new_source_vertex); i_new_edge++) {
						if (new_vertex_edges[i_new_edge] == i_new_target_vertex) {
							already_added = true;
							break;
						}
					}
					//TODO: take care of possible race condition -- adding two equivalent i_new_target_vertex values from separate threads
					if (!already_added) {
						int new_edge_index = new_edge_counts.FetchAdd(i_new_source_vertex, 1);
						if (new_edge_index < new_max_vertex_degree) {
							new_vertex_edges[new_edge_index] = i_new_target_vertex;
						}
					}
				}
			}
	);

}


} // namespace nnrt::geometry::functional::kernel::sampling