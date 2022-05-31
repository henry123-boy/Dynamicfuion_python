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
#include "geometry/kernel/PointDownsampling.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::downsampling {

void DownsamplePointsByRadius(o3c::Tensor& downsampled_points, const o3c::Tensor& original_points, float radius) {
	open3d::utility::LogError("Not fully implemented (see TODO in PointDownsamplingImpl.h)");
	core::InferDeviceFromEntityAndExecute(
			original_points,
			[&] { DownsamplePointsByRadius<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, radius); },
			[&] { NNRT_IF_CUDA(DownsamplePointsByRadius<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, radius);); }
	);
}

void GridDownsamplePoints(open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size) {
	core::InferDeviceFromEntityAndExecute(
			original_points,
			[&] { GridDownsamplePoints<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, grid_cell_size); },
			[&] { NNRT_IF_CUDA(GridDownsamplePoints<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, grid_cell_size);); }
	);
}

} // nnrt::geometry::kernel::downsampling