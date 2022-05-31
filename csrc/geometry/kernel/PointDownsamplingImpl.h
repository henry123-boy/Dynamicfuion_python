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

#include <open3d/core/hashmap/HashMap.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>

// local
#include "geometry/kernel/PointDownsampling.h"
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentAtomics.h"
#include "core/kernel/HashTableUtilities.h"

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;
namespace hash_table = nnrt::core::kernel::hash_table;

namespace nnrt::geometry::kernel::downsampling {

namespace {

template<open3d::core::Device::DeviceType DeviceType, typename TPoint>
NNRT_DEVICE_WHEN_CUDACC inline Eigen::Vector3i
DeterminePointsBinCoordinate(const TPoint& point,
                             const Eigen::Vector3f& grid_bin_minimum,
                             float bin_size) {
	return {
			static_cast<int32_t>((point.x() - grid_bin_minimum.x()) / bin_size),
			static_cast<int32_t>((point.y() - grid_bin_minimum.y()) / bin_size),
			static_cast<int32_t>((point.z() - grid_bin_minimum.z()) / bin_size)
	};
}

template<open3d::core::Device::DeviceType DeviceType>
NNRT_DEVICE_WHEN_CUDACC inline int
RavelBinLinearIndex(const Eigen::Vector3i& grid_bin_coordinate,
                    const Eigen::Vector3i& grid_bin_extents) {
	return grid_bin_extents.z() * grid_bin_extents.y() * grid_bin_coordinate.z() +
	       grid_bin_extents.y() * grid_bin_coordinate.y() +
	       grid_bin_coordinate.x();
}

template<open3d::core::Device::DeviceType DeviceType>
NNRT_DEVICE_WHEN_CUDACC inline Eigen::Vector3i
UnravelBinLinearIndex(int bin_linear_index,
                      const Eigen::Vector3i& grid_bin_extents) {
	Eigen::Vector3i bin_coordinate;
	const int yz_layer_count = grid_bin_extents.z() * grid_bin_extents.y();
	bin_coordinate.z() = bin_linear_index / yz_layer_count;
	bin_coordinate.y() = (bin_linear_index % yz_layer_count) / grid_bin_extents.y();
	bin_coordinate.x() = (bin_linear_index % yz_layer_count) % grid_bin_extents.y();
	return bin_coordinate;
}

struct PointAggregationBin {
#ifdef __CUDACC__
	float x;
	float y;
	float z;
	int count;
#else
	std::atomic<float> x;
	std::atomic<float> y;
	std::atomic<float> z;
	std::atomic<int> count;
#endif
	template<o3c::Device::DeviceType DeviceType>
	NNRT_DEVICE_WHEN_CUDACC
	void UpdateWithPointAndCount(const Eigen::Vector3f& point, int _count){
		x = point.x();
		y = point.y();
		z = point.z();
		this->count = _count;
	}
};


template<open3d::core::Device::DeviceType DeviceType>
void FindPointCollectionBounds(Eigen::Vector3f& grid_bin_min_bound, Eigen::Vector3f& grid_bin_max_bound,
                               const o3gk::NDArrayIndexer& original_point_indexer, const o3c::Device& device,
                               int64_t original_point_count) {
	NNRT_DECLARE_ATOMIC(float, min_x);
	NNRT_INITIALIZE_ATOMIC(float, min_x, 0.f);
	NNRT_DECLARE_ATOMIC(float, min_y);
	NNRT_INITIALIZE_ATOMIC(float, min_y, 0.f);
	NNRT_DECLARE_ATOMIC(float, min_z);
	NNRT_INITIALIZE_ATOMIC(float, min_z, 0.f);
	NNRT_DECLARE_ATOMIC(float, max_x);
	NNRT_INITIALIZE_ATOMIC(float, max_x, 0.f);
	NNRT_DECLARE_ATOMIC(float, max_y);
	NNRT_INITIALIZE_ATOMIC(float, max_y, 0.f);
	NNRT_DECLARE_ATOMIC(float, max_z);
	NNRT_INITIALIZE_ATOMIC(float, max_z, 0.f);

	o3c::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
				NNRT_ATOMIC_MIN(min_x, point.x());
				NNRT_ATOMIC_MIN(min_y, point.y());
				NNRT_ATOMIC_MIN(min_z, point.z());
				NNRT_ATOMIC_MAX(max_x, point.x());
				NNRT_ATOMIC_MAX(max_y, point.y());
				NNRT_ATOMIC_MAX(max_z, point.z());
			}
	);

	grid_bin_min_bound = Eigen::Vector3f(NNRT_GET_ATOMIC_VALUE_CPU(min_x), NNRT_GET_ATOMIC_VALUE_CPU(min_y), NNRT_GET_ATOMIC_VALUE_CPU(min_z));
	grid_bin_max_bound = Eigen::Vector3f(NNRT_GET_ATOMIC_VALUE_CPU(max_x), NNRT_GET_ATOMIC_VALUE_CPU(max_y), NNRT_GET_ATOMIC_VALUE_CPU(max_z));

	NNRT_CLEAN_UP_ATOMIC(min_x);NNRT_CLEAN_UP_ATOMIC(min_y);NNRT_CLEAN_UP_ATOMIC(min_z);NNRT_CLEAN_UP_ATOMIC(max_x);NNRT_CLEAN_UP_ATOMIC(
			max_y);NNRT_CLEAN_UP_ATOMIC(max_z);
}

template<open3d::core::Device::DeviceType DeviceType>
void InitializeBins(const o3c::Device& device,
                    int64_t bin_count,
                    PointAggregationBin* bins) {
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAggregationBin& bin = bins[workload_idx];
				bin.x = 0.f;
				bin.y = 0.f;
				bin.z = 0.f;
				bin.count = 0;
			}
	);
}

template<open3d::core::Device::DeviceType DeviceType>
void ComputeBinAggregates(const o3gk::NDArrayIndexer& original_point_indexer,
                          const o3c::Device& device,
                          int64_t original_point_count,
                          PointAggregationBin* bins,
                          const Eigen::Vector3f& grid_bin_min_bound,
                          const Eigen::Vector3i& grid_bin_extents,
                          float bin_size) {
	o3c::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
				auto bin_coordinate = DeterminePointsBinCoordinate<DeviceType>(point, grid_bin_min_bound, bin_size);
				auto bin_index = RavelBinLinearIndex<DeviceType>(bin_coordinate, grid_bin_extents);
				PointAggregationBin& bin = bins[bin_index];

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
void ComputeBinAverages(int64_t& downsampled_point_count,
                        o3c::Device& device,
                        int64_t bin_count,
                        PointAggregationBin* bins) {
	NNRT_DECLARE_ATOMIC(uint32_t, downsampled_point_count_atomic);
	NNRT_INITIALIZE_ATOMIC(uint32_t, downsampled_point_count_atomic, 0);

	// COMPUTE BIN AVERAGES AND COUNT RESULTING POINTS
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAggregationBin& bin = bins[workload_idx];
#ifdef __CUDACC__
				if(bin.count != 0){
					bin.x /= static_cast<float>(bin.count);
					bin.y /= static_cast<float>(bin.count);
					bin.z /= static_cast<float>(bin.count);
					NNRT_ATOMIC_ADD(downsampled_point_count_atomic, 1L);
				}
#else
				int bin_point_count = bin.count.load();
				if (bin_point_count != 0) {
					bin.x.store(bin.x.load() / static_cast<float>(bin_point_count));
					bin.y.store(bin.y.load() / static_cast<float>(bin_point_count));
					bin.z.store(bin.z.load() / static_cast<float>(bin_point_count));
					downsampled_point_count_atomic++;
				}
#endif
			}
	);

	downsampled_point_count = static_cast<int64_t>(NNRT_GET_ATOMIC_VALUE_CPU(downsampled_point_count_atomic));
}

template<o3c::Device::DeviceType DeviceType>
void ComputeBinAverages(o3c::Device& device,
                        int64_t bin_count,
                        PointAggregationBin* bins) {
	int64_t downsampled_point_count;
	ComputeBinAverages<DeviceType>(downsampled_point_count, device, bin_count, bins);
}





} // anonymous namespace

template<open3d::core::Device::DeviceType DeviceType>
void DownsamplePointsByRadius(o3c::Tensor& downsampled_points, const o3c::Tensor& original_points, float radius) {
	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();
	auto device = original_points.GetDevice();

	Eigen::Vector3f grid_bin_min_bound, grid_bin_max_bound;
	FindPointCollectionBounds<DeviceType>(grid_bin_min_bound, grid_bin_max_bound, original_point_indexer, device, original_point_count);

	// region ESTIMATE BIN COUNT IN DENSE 3D BIN BLOCK & FIND CENTER

	const auto bin_size = radius;

	auto grid_spatial_extents = grid_bin_max_bound - grid_bin_min_bound;
	auto grid_center = grid_bin_min_bound + (grid_spatial_extents / 2.f);

	auto grid_bin_extents = Eigen::Vector3i(std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.x() / bin_size)), 1),
	                                        std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.y() / bin_size)), 1),
	                                        std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.z() / bin_size)), 1));

	// ensure each dimension is divisible by 2, so that later we can process grid directionally with step 2
	grid_bin_extents.x() = grid_bin_extents.x() + grid_bin_extents.x() % 2;
	grid_bin_extents.y() = grid_bin_extents.y() + grid_bin_extents.y() % 2;
	grid_bin_extents.z() = grid_bin_extents.z() + grid_bin_extents.z() % 2;

	const auto bin_count = grid_bin_extents.x() * grid_bin_extents.y() * grid_bin_extents.z();
	// endregion


	o3c::Blob bin_blob(static_cast<int64_t>(sizeof(PointAggregationBin) * bin_count), device);
	auto bins = reinterpret_cast<PointAggregationBin*>(bin_blob.GetDataPtr());
	InitializeBins<DeviceType>(device, bin_count, bins);


	ComputeBinAggregates<DeviceType>(original_point_indexer, device, original_point_count, bins,
	                                 grid_bin_min_bound, grid_bin_extents, bin_size);

	ComputeBinAverages<DeviceType>(device, bin_count, bins);

	//TODO  MERGE BINS AS NECESSARY -- there are some interesting ways to do this on a per-direction basis, but most likely will need to be run twice
	// for some directions

	o3c::ParallelFor(
			device, bin_count / 2,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t bin_index = workload_idx * 2;
				if (bin_index >= bin_count)
					return;
				PointAggregationBin& bin = bins[bin_index];

				bool has_point = false;
				Eigen::Vector3f bin_point;
				int bin_point_count;

				if(bin.count != 0){
					has_point = true;
					bin_point.x() = bin.x;
					bin_point.y() = bin.y;
					bin_point.z() = bin.z;
					bin_point_count = bin.count;
				}

				if (!has_point)
					return;

				auto bin_coordinate = UnravelBinLinearIndex<DeviceType>(workload_idx * 2, grid_bin_extents);
				auto next_bin_coordinate = bin_coordinate + Eigen::Vector3i(1, 0, 0);
				auto next_bin_index = RavelBinLinearIndex<DeviceType>(next_bin_coordinate, grid_bin_extents);
				if (next_bin_index >= bin_count)
					return;
				PointAggregationBin& next_bin = bins[next_bin_index];
				Eigen::Vector3f next_bin_point;
				int next_bin_point_count;
				next_bin_point_count = next_bin.count;

				if(next_bin.count != 0){
					has_point = true;
					next_bin_point.x() = next_bin.x;
					next_bin_point.y() = next_bin.y;
					next_bin_point.z() = next_bin.z;
				}

				if (!has_point)
					return;

				// merge points
				if ((next_bin_point - bin_point).norm() < radius) {
					int new_count = bin_point_count + next_bin_point_count;
					auto average_point =
							(bin_point * bin_point_count + next_bin_point * next_bin_point_count) /
							(new_count);
					auto merged_bin_coordinate = DeterminePointsBinCoordinate<DeviceType>(average_point, grid_bin_min_bound, bin_size);
					auto merged_bin_index = RavelBinLinearIndex<DeviceType>(bin_coordinate, grid_bin_extents);
					if (merged_bin_index == bin_index) {
						bin.template UpdateWithPointAndCount<DeviceType>(average_point, new_count);
						next_bin.count = 0;
					} else {
						next_bin.template UpdateWithPointAndCount<DeviceType>(average_point, new_count);
						bin.count = 0;
					}
				}

			}
	);


}

template<open3d::core::Device::DeviceType DeviceType>
void GridDownsamplePoints(open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size) {
	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();
	auto device = original_points.GetDevice();

	Eigen::Vector3f grid_bin_min_bound, grid_bin_max_bound;
	FindPointCollectionBounds<DeviceType>(grid_bin_min_bound, grid_bin_max_bound, original_point_indexer, device, original_point_count);

	// region ESTIMATE BIN COUNT IN DENSE 3D BIN BLOCK & FIND CENTER

	const auto bin_size = grid_cell_size;

	auto grid_spatial_extents = grid_bin_max_bound - grid_bin_min_bound;
	auto grid_center = grid_bin_min_bound + (grid_spatial_extents / 2.f);

	auto grid_bin_extents = Eigen::Vector3i(std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.x() / bin_size)), 1),
	                                        std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.y() / bin_size)), 1),
	                                        std::max(static_cast<int32_t>(std::ceil(grid_spatial_extents.z() / bin_size)), 1));

	const auto bin_count = grid_bin_extents.x() * grid_bin_extents.y() * grid_bin_extents.z();
	// endregion

	// INITIALIZE BINS
	o3c::Blob bin_blob(static_cast<int64_t>(sizeof(PointAggregationBin) * bin_count), device);
	auto bins = reinterpret_cast<PointAggregationBin*>(bin_blob.GetDataPtr());

	InitializeBins<DeviceType>(device, bin_count, bins);
	ComputeBinAggregates<DeviceType>(original_point_indexer, device, original_point_count, bins,
	                                 grid_bin_min_bound, grid_bin_extents, bin_size);

	int64_t downsampled_point_count;
	ComputeBinAverages<DeviceType>(downsampled_point_count, device, bin_count, bins);

	downsampled_points = o3c::Tensor({downsampled_point_count, 3}, o3c::Float32, device);
	o3gk::NDArrayIndexer downsampled_point_indexer(downsampled_points, 1);

	NNRT_DECLARE_ATOMIC(uint32_t, downsampled_point_count_atomic);
	NNRT_INITIALIZE_ATOMIC(unsigned int, downsampled_point_count_atomic, 0);
	// STORE POINTS IN RESULT TENSOR
	o3c::ParallelFor(
			device, bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				PointAggregationBin& bin = bins[workload_idx];
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


} // namespace nnrt::geometry::kernel::downsampling