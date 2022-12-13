//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/16/22.
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

// open3d
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/core/Dispatch.h>

// local
#include "core/functional/Masking.h"
#include "DeformableMeshRenderToRgbdImageFitter.h"
#include "rendering/RasterizeMesh.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

void DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field, const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::Image& reference_color_image,
        const open3d::t::geometry::PointCloud& reference_point_cloud,
        const open3d::core::Tensor& intrinsic_matrix,
        const open3d::core::Tensor& extrinsic_matrix
) const {
    auto [anchors, weights] = warp_field.PrecomputeAnchorsAndWeights(canonical_mesh,
                                                                     nnrt::geometry::AnchorComputationMethod::SHORTEST_PATH);
    //TODO finish impl
    utility::LogError("Not Implemented");
}

void
DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field, const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::Image& reference_color_image,
        const open3d::t::geometry::Image& reference_depth_image,
        const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
        const open3d::core::Tensor& intrinsic_matrix, const open3d::core::Tensor& extrinsic_matrix,
        float depth_scale, float depth_max
) const {
    open3d::t::geometry::PointCloud point_cloud;
    if (reference_image_mask.has_value()) {
        DISPATCH_DTYPE_TO_TEMPLATE(reference_depth_image.GetDtype(), [&]() {
            auto depth_tensor = reference_depth_image.AsTensor();
            auto masked_reference_depth_image = o3tg::Image(
                    core::functional::SetMaskedToValue(depth_tensor, reference_image_mask.value().get(),
                                                       static_cast<scalar_t>(0.0))
            );
            point_cloud =
                    o3tg::PointCloud::CreateFromDepthImage(masked_reference_depth_image, intrinsic_matrix,
                                                           o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")),
                                                           depth_scale, depth_max);

        });
    } else {
        point_cloud =
                o3tg::PointCloud::CreateFromDepthImage(reference_depth_image, intrinsic_matrix,
                                                       o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")),
                                                       depth_scale, depth_max);
    }
    FitToImage(warp_field, canonical_mesh, reference_color_image, point_cloud, intrinsic_matrix, extrinsic_matrix);
}

void
DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field, const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::RGBDImage& reference_image,
        const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
        const open3d::core::Tensor& intrinsic_matrix, const open3d::core::Tensor& extrinsic_matrix,
        float depth_scale, float depth_max
) const {
    FitToImage(warp_field, canonical_mesh, reference_image.color_, reference_image.depth_, reference_image_mask,
               intrinsic_matrix, extrinsic_matrix,
               depth_scale, depth_max);
}


} // namespace nnrt::alignment