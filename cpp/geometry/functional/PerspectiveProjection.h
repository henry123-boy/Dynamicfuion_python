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
#pragma once

#include <open3d/core/Tensor.h>
#include "../../../../Open3D/Open3D0151/install/include/open3d/t/geometry/Image.h"

// candidate to be moved to a separate "camera" module
namespace nnrt::geometry::functional {

void UnprojectDepthImageWithoutFiltering(
        open3d::core::Tensor& unprojected_points, open3d::core::Tensor& mask, const open3d::t::geometry::Image& depth,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float32,
                                                                           open3d::core::Device("CPU:0")),
        float depth_scale = 1000.0f, float depth_max = 3.0f, bool preserve_pixel_layout = false
);

open3d::core::Tensor UnprojectProjectedPoints(
        const open3d::core::Tensor& projected_points, const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics =
                open3d::core::Tensor::Eye(4, open3d::core::Float32, open3d::core::Device("CPU:0")),
        float depth_scale = 1.0f, float depth_max = 20.0f
);

void UnprojectProjectedPoints(
        open3d::core::Tensor& unprojected_points,
        const open3d::core::Tensor& projected_points, const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics =
                open3d::core::Tensor::Eye(4, open3d::core::Float32, open3d::core::Device("CPU:0")),
        float depth_scale = 1.0f, float depth_max = 20.0f
);


} // namespace nnrt::geometry::functional
