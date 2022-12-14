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
#include "PointCloud.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace nnrt::geometry::functional::kernel{

void UnprojectWithoutDepthFiltering(
		open3d::core::Tensor& points, open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> colors,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> mask,
		const open3d::core::Tensor& depth, const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& image_colors,
		const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
) {
	core::ExecuteOnDevice(
			depth.GetDevice(),
			[&] {
				UnprojectWithoutDepthFiltering < o3c::Device::DeviceType::CPU > (points, colors, mask, depth, image_colors, intrinsics,
						extrinsics, depth_scale, depth_max, preserve_image_layout);
			},
			[&] {
				NNRT_IF_CUDA(
						UnprojectWithoutDepthFiltering < o3c::Device::DeviceType::CUDA > (points, colors, mask, depth, image_colors, intrinsics,
								extrinsics, depth_scale, depth_max, preserve_image_layout);
				);
			}
	);
}
} // namespace nnrt::geometry::functional::kernel