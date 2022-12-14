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
#include "geometry/functional/kernel/PerspectiveProjectionImpl.h"

namespace nnrt::geometry::functional::kernel {
template void UnprojectRasterWithoutDepthFiltering<o3c::Device::DeviceType::CPU>(
        open3d::core::Tensor& unprojected_points, open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> unprojected_point_colors,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> depth_mask,
        const open3d::core::Tensor& raster_depth, open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> raster_colors,
        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
);

template
void Unproject<open3d::core::Device::DeviceType::CPU>(
        open3d::core::Tensor& unprojected_points,
        const open3d::core::Tensor& projected_points,
        const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics, // assumes checked
        const open3d::core::Tensor& extrinsics, // assumes checked
        float depth_scale,
        float depth_max
);

} // namespace nnrt::geometry::functional::kernel