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
#include "geometry/kernel/RadiusDownsampling.h"
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentAtomics.h"
#include "core/kernel/HashTableUtilities.h"

namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;
namespace hash_table = nnrt::core::kernel::hash_table;

namespace nnrt::geometry::kernel::downsampling {

template<open3d::core::Device::DeviceType DeviceType>
void DownsamplePointsByRadius(o3c::Tensor& downsampled_points, const o3c::Tensor& original_points, float radius) {
	o3gk::NDArrayIndexer original_point_indexer(original_points, 1);
	auto original_point_count = original_points.GetLength();
	auto device = original_points.GetDevice();

	//region GET POINT COLLECTION BOUNDS / EXTREMA

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
				Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
				NNRT_ATOMIC_MIN(min_x, point.x());
				NNRT_ATOMIC_MIN(min_y, point.y());
				NNRT_ATOMIC_MIN(min_z, point.z());
				NNRT_ATOMIC_MAX(max_x, point.x());
				NNRT_ATOMIC_MAX(max_y, point.y());
				NNRT_ATOMIC_MAX(max_z, point.z());
			}
	);

	Eigen::Vector3f min_bound(NNRT_GET_ATOMIC_VALUE_CPU(min_x), NNRT_GET_ATOMIC_VALUE_CPU(min_y), NNRT_GET_ATOMIC_VALUE_CPU(min_z));
	Eigen::Vector3f max_bound(NNRT_GET_ATOMIC_VALUE_CPU(max_x), NNRT_GET_ATOMIC_VALUE_CPU(max_y), NNRT_GET_ATOMIC_VALUE_CPU(max_z));

	NNRT_CLEAN_UP_ATOMIC(min_x);NNRT_CLEAN_UP_ATOMIC(min_y);NNRT_CLEAN_UP_ATOMIC(min_z);NNRT_CLEAN_UP_ATOMIC(max_x);NNRT_CLEAN_UP_ATOMIC(
			max_y);NNRT_CLEAN_UP_ATOMIC(max_z);
	//endregion

	// region ESTIMATE BIN COUNT & FIND CENTER
	//TODO: can't use pi C++20's <numbers> header due to lack of C++20 support in CUDA (see https://stackoverflow.com/a/57285400/844728a)
	const double angle = M_PI * (45. / 180.);
	const auto bin_size = static_cast<float>(2.0 * radius * std::cos(angle));
	auto grid_extents = max_bound - min_bound;
	auto grid_center = min_bound + (grid_extents / 2.f);
	//@formatter:off
	const auto max_bin_count =
			static_cast<int32_t>(
			std::ceil(grid_extents.x() / bin_size) + 1 *
			std::ceil(grid_extents.y() / bin_size) + 1 *
			std::ceil(grid_extents.z() / bin_size) + 1
	);
	//@formatter:on

	// constexpr float estimated_bin_density = 0.5;
	// const auto bin_count = static_cast<int64_t> (max_bin_count * estimated_bin_density);
	const auto ordered_bin_count = hash_table::ClosestLeastPowerOf2<int32_t>(max_bin_count);
	const auto hash_mask = ordered_bin_count - 1;
	const auto excess_bin_count = 0x200;
	const auto hash_bin_count = ordered_bin_count + excess_bin_count;
	int last_free_excess_index = excess_bin_count - 1;
	// endregion

	// region DETERMINE BIN FOR EACH POINT
	// o3c::HashMap average_hash_map(bin_count, o3c::Int16, o3c::SizeVector{3}, o3c::Float32, o3c::SizeVector{4},
	//                               original_points.GetDevice(), o3c::HashBackendType::Default);
	//o3c::Tensor original_point_bin_keys({original_point_count}, o3c::Int32, device);
	//o3gk::NDArrayIndexer original_point_bin_key_indexer(original_point_bin_keys, 1);
	//o3c::Tensor original_point_bin_coordinates({original_point_count, 3}, o3c::Int16, device);
	// o3gk::NDArrayIndexer original_point_bin_key_indexer(original_point_bin_indices, 1);

	o3c::Blob hash_table_blob(sizeof(hash_table::AveragePointHashBin) * hash_bin_count, device);
	auto hash_table = reinterpret_cast<hash_table::AveragePointHashBin*>(hash_table_blob.GetDataPtr());

	o3c::Tensor utilized_hash_bins({hash_bin_count}, o3c::Int32, device);
	auto utilized_hash_bin_data = utilized_hash_bins.GetDataPtr<int32_t>();

	o3c::ParallelFor(
			device, hash_bin_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				hash_table::AveragePointHashBin& bin = hash_table[workload_idx];
				bin.active = false;
				NNRT_INITIALIZE_ATOMIC(int, bin.count, 0);
				NNRT_INITIALIZE_ATOMIC(float, bin.x, 0.f);
				NNRT_INITIALIZE_ATOMIC(float, bin.y, 0.f);
				NNRT_INITIALIZE_ATOMIC(float, bin.z, 0.f);
			}
	);

	o3c::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto original_point = Eigen::Map<Eigen::Vector3f>(original_point_indexer.GetDataPtr<float>(workload_idx));
				auto block_coordinates = hash_table::DeterminePointsBinCoordinate<DeviceType>(original_point, grid_center, bin_size);
				int hash_code = hash_table::FindHashCodeAt<DeviceType>(hash_table, block_coordinates, hash_mask, ordered_bin_count);
				if(hash_code == -1){

				}

			}
	);
	// endregion


}

} // namespace nnrt::geometry::kernel::downsampling