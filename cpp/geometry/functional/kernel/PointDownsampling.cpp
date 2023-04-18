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
#include "PointDownsampling.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::functional::kernel::downsampling {


void GridDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend
) {
	core::ExecuteOnDevice(
			original_points.GetDevice(),
			[&] {
				GridDownsamplePoints<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, grid_cell_size, hash_backend);
			},
			[&] {
				NNRT_IF_CUDA(
						GridDownsamplePoints<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, grid_cell_size,
						                                                    hash_backend);
				);
			}
	);
}

void RadiusDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float radius,
		const open3d::core::HashBackendType& hash_backend
) {
	core::ExecuteOnDevice(
			original_points.GetDevice(),
			[&] {
				RadiusDownsamplePoints<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, radius, hash_backend);
			},
			[&] {
				NNRT_IF_CUDA(
						RadiusDownsamplePoints<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, radius,
						                                                      hash_backend);
				);
			}
	);
}

} // nnrt::geometry::functional::kernel::downsampling