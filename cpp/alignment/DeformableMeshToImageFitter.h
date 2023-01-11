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
#pragma once


// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/RGBDImage.h>

// local
#include "../geometry/GraphWarpField.h"

namespace nnrt::alignment {

class DeformableMeshToImageFitter {
public:
    DeformableMeshToImageFitter(
            int maximal_iteration_count = 100,
            float minimal_update_threshold = 1e-6,
            bool use_perspective_correction = false,
            float max_depth = 10.f
    );

    /**
     * \brief
     * \param warp_field
     * \param canonical_mesh
     * \param reference_image
     * \param intrinsic_matrix
     * \param extrinsic_matrix -- Note: has nothing to do with the reference RGBD image, which is assumed to have identity extrinsics (rel. to camera)
     * \param depth_scale
     * \param depth_max
     */
    void FitToImage(
            nnrt::geometry::GraphWarpField& warp_field,
            const open3d::t::geometry::TriangleMesh& canonical_mesh,
            const open3d::t::geometry::RGBDImage& reference_image,
            const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
            const open3d::core::Tensor& intrinsic_matrix,
            const open3d::core::Tensor& extrinsic_matrix,
            float depth_scale
    )
    const;

    void FitToImage(
            nnrt::geometry::GraphWarpField& warp_field,
            const open3d::t::geometry::TriangleMesh& canonical_mesh,
            const open3d::t::geometry::Image& reference_color_image,
            const open3d::t::geometry::Image& reference_depth_image,
            const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
            const open3d::core::Tensor& intrinsic_matrix,
            const open3d::core::Tensor& extrinsic_matrix,
            float depth_scale
    )
    const;

    void FitToImage(
            nnrt::geometry::GraphWarpField& warp_field,
            const open3d::t::geometry::TriangleMesh& canonical_mesh,
            const open3d::t::geometry::Image& reference_color_image,
            const open3d::t::geometry::PointCloud& reference_point_cloud,
            const open3d::core::Tensor& reference_point_mask,
            const open3d::core::Tensor& intrinsic_matrix,
            const open3d::core::Tensor& extrinsic_matrix
    )
    const;


private:
    int max_iteration_count;
    float min_update_threshold;
    float max_depth;
    bool use_perspective_correction;

    open3d::core::Tensor ComputeResiduals(
            open3d::t::geometry::PointCloud& rasterized_point_cloud,
            open3d::core::Tensor& residual_mask,
            const open3d::t::geometry::TriangleMesh& warped_mesh,
            const open3d::core::Tensor& pixel_face_indices,
            const open3d::core::Tensor& pixel_barycentric_coordinates,
            const open3d::core::Tensor& pixel_depths,
            const open3d::t::geometry::Image& reference_color_image,
            const open3d::t::geometry::PointCloud& reference_point_cloud,
            const open3d::core::Tensor& reference_point_mask,
            const open3d::core::Tensor& intrinsics
    )
    const;

    open3d::core::Tensor ComputeHessianApproximation_BlockDiagonal(
            const open3d::t::geometry::PointCloud& rasterized_point_cloud,
            const open3d::t::geometry::PointCloud& reference_point_cloud,
            const open3d::t::geometry::TriangleMesh& warped_mesh,
            const open3d::core::Tensor& pixel_faces,
            const open3d::core::Tensor& vertex_anchors,
            const open3d::core::Tensor& residual_mask,
            const open3d::core::Tensor& rasterized_vertex_position_jacobians,
            const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
            const open3d::core::Tensor& warped_vertex_position_jacobians,
            const open3d::core::Tensor& warped_vertex_normal_jacobians,
            int64_t node_count
    )
    const;
};


} // namespace nnrt::alignment
