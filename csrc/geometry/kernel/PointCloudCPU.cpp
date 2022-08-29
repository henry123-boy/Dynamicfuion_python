//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/29/22.
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
#include "geometry/kernel/PointCloudImpl.h"

namespace nnrt::geometry::kernel::pointcloud {
template void UnprojectWithoutDepthFiltering<o3c::Device::DeviceType::CPU>(
		o3c::Tensor& points, utility::optional<std::reference_wrapper<o3c::Tensor>> colors,
		utility::optional<std::reference_wrapper<o3c::Tensor>> mask,
		const o3c::Tensor& depth, utility::optional<std::reference_wrapper<const o3c::Tensor>> image_colors,
		const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
);
} // namespace nnrt::geometry::kernel::pointcloud