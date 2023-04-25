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


template<open3d::core::Device::DeviceType DeviceType>
void AveragePointsIntoGridCells(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	o3c::Tensor grid_offset_tensor(std::vector<float>{grid_offset.x(), grid_offset.y(), grid_offset.z()}, {1, 3}, o3c::Float32, device);

	o3c::Tensor point_bins_float = original_points / grid_cell_size + grid_offset_tensor;
	o3c::Tensor point_bins_integer = point_bins_float.Floor().To(o3c::Int32);

	o3c::HashMap point_bin_coord_map(point_bins_integer.GetLength(), o3c::Int32, {3}, o3c::UInt8, {sizeof(PointAverageAggregationBin)}, device,
	                                 hash_backend);

	// activate entries in the hash map that correspond to the
	o3c::Tensor bin_indices_tensor, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, bin_indices_tensor, success_mask);
	o3c::Tensor bins_integer = point_bins_integer.GetItem(o3c::TensorKey::IndexTensor(success_mask));
	auto bin_count = bins_integer.GetLength();

	PointAverageAggregationBin* bins = reinterpret_cast<PointAverageAggregationBin*>(point_bin_coord_map.GetValueTensor().GetDataPtr());
	InitializeAveragingBins<DeviceType>(device, bin_count, bins);

	std::tie(bin_indices_tensor, success_mask) = point_bin_coord_map.Find(point_bins_integer);
	auto bin_indices = bin_indices_tensor.GetDataPtr<int32_t>();

	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();

	ComputeBinAverageAggregates<DeviceType>(original_point_indexer, device, original_point_count, bins, bin_indices);
	ComputeBinAverages<DeviceType>(device, bin_count, bins);

	downsampled_points = o3c::Tensor({bin_count, 3}, o3c::Float32, device);
	o3gk::NDArrayIndexer downsampled_point_indexer(downsampled_points, 1);

	TransferFromBinsToTensor_NoCountCheck<DeviceType>(downsampled_point_indexer, device, bin_count, bins);
}

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

template<open3d::core::Device::DeviceType TDeviceType>
void ExclusivePrefixSum(o3c::Tensor& prefix_sum, const o3c::Tensor& source);

template<>
void ExclusivePrefixSum<open3d::core::Device::DeviceType::CPU>(o3c::Tensor& prefix_sum, const o3c::Tensor& source){

}

template<open3d::core::Device::DeviceType TDeviceType>
void ComputeDistanceMatrices(
		o3c::Tensor& distance_matrices, o3c::Device& device, const o3c::Tensor& bin_indices,
		const o3c::Tensor& bin_point_counts, const o3c::Tensor& points, int64_t distance_matrix_element_count
) {
	auto distance_matrix_data = distance_matrices.GetDataPtr<float>();

	auto bin_index_data = bin_indices.GetDataPtr<int32_t>();
	auto bin_count_data = bin_point_counts.GetDataPtr<int32_t>();
	auto point_data = points.GetDataPtr<float>();


	//TODO
	// o3c::ParallelFor(
	// 		device, distance_matrix_element_count,
	// 		NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_element) {
	// 			int32_t i_bin = bin_index_data[i_node];
	// 			bin_point_counts.FetchAdd(i_bin, 1);
	// 		}
	// );
	// return bin_point_counts.AsTensor(true);
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
	o3c::Tensor bin_indices_tensor, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, bin_indices_tensor, success_mask);
	o3c::Tensor bins_integer = point_bins_integer.GetItem(o3c::TensorKey::IndexTensor(success_mask));
	auto bin_count = bins_integer.GetLength();

	std::tie(bin_indices_tensor, success_mask) = point_bin_coord_map.Find(point_bins_integer);
	auto bin_indices = bin_indices_tensor.GetDataPtr<int32_t>();
	o3c::Tensor bin_point_counts = ComputeBinPointCounts<TDeviceType>(device, bin_indices_tensor, bin_count);
	o3c::Tensor point_counts_squared = bin_point_counts * bin_point_counts;
	int64_t distance_matrix_set_element_count = point_counts_squared.Sum({0}).ToFlatVector<int32_t>()[0];
	// represent a ragged 3D tensor using a 1D tensor
	o3c::Tensor distance_matrices({distance_matrix_set_element_count}, o3c::Float32, device);


	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();

	//TODO finish
	utility::LogError("Not implemented");
	// ComputeBinAverageAggregates<TDeviceType>(original_point_indexer, device, original_point_count, bins, bin_indices);
	// ComputeBinAverages<TDeviceType>(device, bin_count, bins);
	//
	// sample = o3c::Tensor({bin_count}, o3c::Int32, device);
	// o3gk::NDArrayIndexer downsampled_point_indexer(sample, 1);
	//
	// TransferFromBinsToTensor_NoCountCheck<TDeviceType>(downsampled_point_indexer, device, bin_count, bins);
}


template<open3d::core::Device::DeviceType TDeviceType>
void RadiusMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float radius,
		const open3d::core::HashBackendType& hash_backend_type
) {
	float extended_radius = sqrtf(2 * (radius * radius));
	o3c::Tensor downsampled_points_stage_1;
	//TODO: this is not so simple if returning indices instead of points. We'll need to take that into account & change args below.
	MedianGridSamplePoints<TDeviceType>(downsampled_points_stage_1, points, extended_radius * 2, hash_backend_type);
	// merge again while offsetting the grid
	MedianGridSamplePoints<TDeviceType>(sample, downsampled_points_stage_1, extended_radius * 2, hash_backend_type, Eigen::Vector3f(0.5, 0.5, 0.5));
}


} // namespace nnrt::geometry::functional::kernel::sampling