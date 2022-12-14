//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/18/22.
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

// 3rd party
#include <open3d/core/Device.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>


// local
#include "core/PlatformIndependentQualifiers.h"
#include "core/PlatformIndependentAtomics.h"


namespace nnrt::geometry::kernel {
template<open3d::core::Device::DeviceType DeviceType>
void FindPointCollectionBounds(Eigen::Vector3f& grid_bin_min_bound, Eigen::Vector3f& grid_bin_max_bound,
                               const open3d::t::geometry::kernel::NDArrayIndexer& original_point_indexer, const open3d::core::Device& device,
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

	open3d::core::ParallelFor(
			device, original_point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE
					NNRT_DEVICE_WHEN_CUDACC(int64_t
					                        workload_idx) {
				const Eigen::Map<Eigen::Vector3f> point(original_point_indexer.GetDataPtr<float>(workload_idx));
				NNRT_ATOMIC_MIN(min_x, point.x());
				NNRT_ATOMIC_MIN(min_y, point.y());
				NNRT_ATOMIC_MIN(min_z, point.z());
				NNRT_ATOMIC_MAX(max_x, point.x());
				NNRT_ATOMIC_MAX(max_y, point.y());
				NNRT_ATOMIC_MAX(max_z, point.z());
			}
	);

	grid_bin_min_bound = Eigen::Vector3f(NNRT_GET_ATOMIC_VALUE_HOST(min_x), NNRT_GET_ATOMIC_VALUE_HOST(min_y), NNRT_GET_ATOMIC_VALUE_HOST(min_z));
	grid_bin_max_bound = Eigen::Vector3f(NNRT_GET_ATOMIC_VALUE_HOST(max_x), NNRT_GET_ATOMIC_VALUE_HOST(max_y), NNRT_GET_ATOMIC_VALUE_HOST(max_z));

	NNRT_CLEAN_UP_ATOMIC(min_x);NNRT_CLEAN_UP_ATOMIC(min_y);NNRT_CLEAN_UP_ATOMIC(min_z);NNRT_CLEAN_UP_ATOMIC(max_x);NNRT_CLEAN_UP_ATOMIC(
			max_y);NNRT_CLEAN_UP_ATOMIC(max_z);
}

} // namespace nnrt::geometry::kernel {