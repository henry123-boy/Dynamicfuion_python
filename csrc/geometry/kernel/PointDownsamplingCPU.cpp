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
#include "geometry/kernel/PointDownsamplingImpl.h"

namespace nnrt::geometry::kernel::downsampling {

template
void DownsamplePointsByRadius<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float radius
);

template
void GridDownsamplePoints<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size
);

} // namespace nnrt::geometry::kernel::downsampling