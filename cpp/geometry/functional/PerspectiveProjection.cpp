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
#include <open3d/t/geometry/Utility.h>

#include "PerspectiveProjection.h"
#include "geometry/functional/kernel/PerspectiveProjection.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace utility = open3d::utility;

namespace nnrt::geometry::functional {
void UnprojectDepthImageWithoutFiltering(
        open3d::core::Tensor& unprojected_points, open3d::core::Tensor& mask, const open3d::t::geometry::Image& depth,
        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, float depth_scale,
        float depth_max, bool preserve_pixel_layout
) {

    o3tg::CheckIntrinsicTensor(intrinsics);
    o3tg::CheckExtrinsicTensor(extrinsics);

    kernel::UnprojectRasterWithoutDepthFiltering(
            unprojected_points, utility::nullopt, mask, depth.AsTensor(), utility::nullopt, intrinsics,
            extrinsics, depth_scale, depth_max, preserve_pixel_layout
    );
}

void UnprojectProjectedPoints(
        open3d::core::Tensor& unprojected_points,
        const open3d::core::Tensor& projected_points,
        const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics,
        float depth_scale,
        float depth_max
) {
    o3tg::CheckIntrinsicTensor(intrinsics);
    o3tg::CheckExtrinsicTensor(extrinsics);
    kernel::Unproject(unprojected_points, projected_points, projected_point_depth, intrinsics, extrinsics, depth_scale,
                      depth_max);
}

open3d::core::Tensor UnprojectProjectedPoints(
        const open3d::core::Tensor& projected_points,
        const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics,
        float depth_scale, float depth_max
) {
    open3d::core::Tensor unprojected_points;
    UnprojectProjectedPoints(unprojected_points, projected_points, projected_point_depth, intrinsics, extrinsics,
                             depth_scale, depth_max);
    return unprojected_points;
}


} // namespace nnrt::geometry::functional