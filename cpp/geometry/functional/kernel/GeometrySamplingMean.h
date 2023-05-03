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
// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

// local includes
#include "geometry/functional/kernel/PointAggregationBins.h"
#include "geometry/functional/kernel/GeometrySamplingGridBinning.h"


//*** === header for kernel usage only. === ***

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::geometry::functional::kernel::sampling::mean {


template<open3d::core::Device::DeviceType DeviceType>
void InitializeAveragingBins(
		const o3c::Device& device,
		int64_t bin_count,
		PointMeanAggregationBin* bins
) {
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		PointMeanAggregationBin& bin = bins[workload_idx];
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
		PointMeanAggregationBin* bins,
		int32_t* bin_indices
) {
	o3c::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
		auto bin_index = bin_indices[workload_idx];
		PointMeanAggregationBin& bin = bins[bin_index];
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
		PointMeanAggregationBin* bins
) {

	// COMPUTE BIN AVERAGES
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		PointMeanAggregationBin& bin = bins[workload_idx];
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
void TransferFromMeanBinsToTensor_NoCountCheck(
		o3gk::NDArrayIndexer& downsampled_point_indexer,
		o3c::Device& device,
		int64_t bin_count,
		PointMeanAggregationBin* bins
) {

	// STORE POINTS IN RESULT TENSOR
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		PointMeanAggregationBin& bin = bins[workload_idx];
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
void TransferFromMeanBinsToTensor_CountCheck(
		o3gk::NDArrayIndexer& downsampled_point_indexer,
		o3c::Device& device,
		int64_t bin_count,
		PointMeanAggregationBin* bins
) {

	NNRT_DECLARE_ATOMIC(uint32_t, downsampled_point_count_atomic);
	NNRT_INITIALIZE_ATOMIC(unsigned int, downsampled_point_count_atomic, 0);
	// STORE POINTS IN RESULT TENSOR
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
		PointMeanAggregationBin& bin = bins[workload_idx];
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


template<open3d::core::Device::DeviceType TDeviceType>
void ComputeMeanOfPointsInGridCells(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	auto [bin_indices_tensor, bin_count, point_bin_coord_map] =
			GridBinPoints<TDeviceType, PointMeanAggregationBin>(original_points, grid_cell_size, hash_backend, grid_offset);

	auto bin_indices = bin_indices_tensor.template GetDataPtr<int32_t>();
	auto* bins = reinterpret_cast<PointMeanAggregationBin*>(point_bin_coord_map.GetValueTensor().GetDataPtr());
	InitializeAveragingBins<TDeviceType>(device, bin_count, bins);

	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();

	ComputeBinAverageAggregates<TDeviceType>(original_point_indexer, device, original_point_count, bins, bin_indices);
	ComputeBinAverages<TDeviceType>(device, bin_count, bins);

	downsampled_points = o3c::Tensor({bin_count, 3}, o3c::Float32, device);
	o3gk::NDArrayIndexer downsampled_point_indexer(downsampled_points, 1);

	TransferFromMeanBinsToTensor_NoCountCheck<TDeviceType>(downsampled_point_indexer, device, bin_count, bins);
}


} // namespace nnrt::geometry::functional::kernel::sampling::mean
