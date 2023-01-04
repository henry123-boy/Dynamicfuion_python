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
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/Comparison.h"
#include "alignment/functional/WarpedVertexAndNormalJacobians.h"
#include "alignment/functional/RasterizedVertexAndNormalJacobians.h"
#include "rendering/kernel/CoordinateSystemConversions.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

DeformableMeshRenderToRgbdImageFitter::DeformableMeshRenderToRgbdImageFitter(
        int maximal_iteration_count,
        float minimal_update_threshold,
        bool use_perspective_correction,
        float max_depth
) : max_iteration_count(maximal_iteration_count),
    min_update_threshold(minimal_update_threshold),
    use_perspective_correction(use_perspective_correction) {}

void DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field,
        const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::Image& reference_color_image,
        const open3d::t::geometry::PointCloud& reference_point_cloud,
        const open3d::core::Tensor& reference_point_mask,
        const open3d::core::Tensor& intrinsic_matrix,
        const open3d::core::Tensor& extrinsic_matrix
) const {
    auto [warp_anchors, warp_weights] = warp_field.PrecomputeAnchorsAndWeights(canonical_mesh,
                                                                               nnrt::geometry::AnchorComputationMethod::SHORTEST_PATH);
    o3c::SizeVector image_size = {reference_color_image.GetRows(), reference_color_image.GetCols()};

    int iteration = 0;
    float maximum_update = std::numeric_limits<float>::max();

    auto [ndc_intrinsic_matrix, ndc_xy_range] =
            rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsic_matrix, image_size);


    while (iteration < max_iteration_count && maximum_update > min_update_threshold) {
        o3tg::TriangleMesh warped_mesh = warp_field.WarpMesh(warped_mesh, warp_anchors, warp_weights, true, extrinsic_matrix);


        auto [extracted_face_vertices, clipped_face_mask] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(
                    warped_mesh, intrinsic_matrix, image_size, 0.0, 10.0
            );

        //TODO: add an obvious optional optimization in the case perspective-correct barycentrics are used: output the
        // distorted barycentric coordinates and reuse them, instead of recomputing them, in RasterizedVertexAndNormalJacobians
        auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
                nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1,
                                                       -1, -1, this->use_perspective_correction, false, true);

        // compute residuals r
        o3c::Tensor residuals =
                this->ComputeResiduals(warped_mesh, pixel_face_indices, pixel_barycentric_coordinates, pixel_depths,
                                       reference_color_image, reference_point_cloud, reference_point_mask,
                                       intrinsic_matrix);

        // compute warped vertex and normal jacobians wrt. delta rotations and jacobians
        auto [warped_vertex_position_jacobians, warped_vertex_normal_jacobians] =
                functional::WarpedVertexAndNormalJacobians(canonical_mesh, warp_field, warp_anchors, warp_weights);


        // compute rasterized vertex and normal jacobians
        auto [rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians] =
                functional::RasterizedVertexAndNormalJacobians(warped_mesh, pixel_face_indices,
                                                               pixel_barycentric_coordinates,
                                                               ndc_intrinsic_matrix, this->use_perspective_correction);
        // compute (J^T)J
        rendred_
        // compute -Jr

        // solve system of linear equations for the delta rotations and translations

        iteration++;
    }
}

void
DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field,
        const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::Image& reference_color_image,
        const open3d::t::geometry::Image& reference_depth_image,
        const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
        const open3d::core::Tensor& intrinsic_matrix,
        const open3d::core::Tensor& extrinsic_matrix,
        float depth_scale
) const {
    o3c::Tensor identity_extrinsics = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));

    o3c::Tensor points, reference_point_depth_mask;
    nnrt::geometry::functional::UnprojectDepthImageWithoutFiltering(
            points, reference_point_depth_mask, reference_depth_image.AsTensor(), intrinsic_matrix, identity_extrinsics,
            depth_scale, this->max_depth, false
    );

    open3d::t::geometry::PointCloud point_cloud(points);

    o3c::Tensor final_reference_point_mask;
    if (reference_image_mask.has_value()) {
        final_reference_point_mask = reference_point_depth_mask.LogicalOr(
                reference_image_mask.value().get().Reshape(reference_point_depth_mask.GetShape())
        );
    } else {
        final_reference_point_mask = reference_point_depth_mask;
    }

    FitToImage(warp_field, canonical_mesh, reference_color_image, point_cloud, final_reference_point_mask,
               intrinsic_matrix, extrinsic_matrix);
}

void
DeformableMeshRenderToRgbdImageFitter::FitToImage(
        nnrt::geometry::GraphWarpField& warp_field,
        const open3d::t::geometry::TriangleMesh& canonical_mesh,
        const open3d::t::geometry::RGBDImage& reference_image,
        const open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>>& reference_image_mask,
        const open3d::core::Tensor& intrinsic_matrix,
        const open3d::core::Tensor& extrinsic_matrix,
        float depth_scale
) const {
    FitToImage(warp_field, canonical_mesh, reference_image.color_, reference_image.depth_, reference_image_mask,
               intrinsic_matrix, extrinsic_matrix,
               depth_scale);

}

open3d::core::Tensor DeformableMeshRenderToRgbdImageFitter::ComputeResiduals(
        const open3d::t::geometry::TriangleMesh& warped_mesh,
        const open3d::core::Tensor& pixel_face_indices,
        const open3d::core::Tensor& pixel_barycentric_coordinates,
        const open3d::core::Tensor& pixel_depths,
        const open3d::t::geometry::Image& reference_color_image,
        const open3d::t::geometry::PointCloud& reference_point_cloud,
        const open3d::core::Tensor& reference_point_mask,
        const open3d::core::Tensor& intrinsics
) const {

    o3c::SizeVector image_size = {reference_color_image.GetRows(), reference_color_image.GetCols()};

    auto vertex_normals = warped_mesh.GetVertexNormals();
    auto triangle_indices = warped_mesh.GetTriangleIndices();
    auto face_vertex_normals = vertex_normals.GetItem(o3c::TensorKey::IndexTensor(triangle_indices));

    auto rendered_normals =
            nnrt::rendering::functional::InterpolateVertexAttributes(
                    pixel_face_indices, pixel_barycentric_coordinates, face_vertex_normals
            );

    o3c::Tensor identity_extrinsics = o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0"));

    o3c::Tensor rendered_points, rendered_point_mask;
    nnrt::geometry::functional::UnprojectDepthImageWithoutFiltering(
            rendered_points, rendered_point_mask, pixel_depths, intrinsics, identity_extrinsics,
            1.0f, this->max_depth, false);

    o3tg::PointCloud rendered_point_cloud(rendered_points);
    rendered_point_cloud.SetPointNormals(rendered_normals);

    o3c::Tensor distances =
            geometry::functional::ComputePointToPlaneDistances(rendered_point_cloud, reference_point_cloud);

    o3c::Tensor global_point_mask = reference_point_mask.LogicalAnd(rendered_point_mask);

    core::functional::SetMaskedToValue(distances, global_point_mask, 0.0f);

    return distances;
}


} // namespace nnrt::alignment