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
#include <open3d/core/ParallelFor.h>
#include "geometry/functional/kernel/GeometrySamplingImpl.h"

namespace nnrt::geometry::functional::kernel::sampling {

template
void GridDownsamplePoints<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& downsampled_points,
		const open3d::core::Tensor& original_points,
		float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend
);

template
void FastRadiusDownsamplePoints<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& downsampled_points,
		const open3d::core::Tensor& original_points,
		float min_distance,
		const open3d::core::HashBackendType& hash_backend
);

template
void RadiusMedianSubsample3dPoints<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& sample, const open3d::core::Tensor& points, float min_distance,
		const open3d::core::HashBackendType& hash_backend
);

template
void RadiusSubsampleGraph<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& sample,
		open3d::core::Tensor& resampled_edges,
		const open3d::core::Tensor& vertices,
		const open3d::core::Tensor& edges,
		float radius
);

} // namespace nnrt::geometry::functional::kernel::sampling