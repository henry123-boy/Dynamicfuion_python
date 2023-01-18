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
#include "DeformableMeshToImageFitter.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/PointToPlaneDistances.h"
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/functional/ExtractFaceVertices.h"
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "alignment/functional/WarpedVertexAndNormalJacobians.h"
#include "alignment/functional/RasterizedVertexAndNormalJacobians.h"
#include "alignment/kernel/DeformableMeshToImageFitter.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

DeformableMeshToImageFitter::DeformableMeshToImageFitter(
        int maximal_iteration_count,
        float minimal_update_threshold,
        bool use_perspective_correction,
        float max_depth
) : max_iteration_count(maximal_iteration_count),
    min_update_threshold(minimal_update_threshold),
    use_perspective_correction(use_perspective_correction) {}

void DeformableMeshToImageFitter::FitToImage(
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
        o3tg::TriangleMesh
                warped_mesh = warp_field.WarpMesh(warped_mesh, warp_anchors, warp_weights, true, extrinsic_matrix);


        auto [extracted_face_vertices, clipped_face_mask] =
                nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(
                        warped_mesh, intrinsic_matrix, image_size, 0.0, 10.0
                );

        //TODO: add an obvious optional optimization in the case perspective-correct barycentrics are used: output the
        // distorted barycentric coordinates and reuse them, instead of recomputing them, in RasterizedVertexAndNormalJacobians
        auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
                nnrt::rendering::RasterizeNdcTriangles(extracted_face_vertices, clipped_face_mask, image_size, 0.f, 1,
                                                       -1, -1, this->use_perspective_correction, false, true);

        // compute residuals r, retain rasterized points & global mask relevant for energy function being minimized
        o3tg::PointCloud rasterized_point_cloud;
        o3c::Tensor global_point_mask;
        o3c::Tensor residuals =
                this->ComputeResiduals(rasterized_point_cloud, global_point_mask, warped_mesh, pixel_face_indices,
                                       pixel_barycentric_coordinates, pixel_depths,
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


        // compute (J^T)J, i.e. hessian approximation
        //TODO

        // compute -Jr
        // TODO
        // solve system of linear equations for the delta rotations and translations
        // TODO

        iteration++;
    }
}

void
DeformableMeshToImageFitter::FitToImage(
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
DeformableMeshToImageFitter::FitToImage(
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

open3d::core::Tensor DeformableMeshToImageFitter::ComputeResiduals(
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

    o3c::Tensor rasterized_points;
    nnrt::geometry::functional::UnprojectDepthImageWithoutFiltering(
            rasterized_points, residual_mask, pixel_depths, intrinsics, identity_extrinsics,
            1.0f, this->max_depth, false);

    rasterized_point_cloud = o3tg::PointCloud(rasterized_points);
    rasterized_point_cloud.SetPointNormals(rendered_normals);

    o3c::Tensor distances =
            geometry::functional::ComputePointToPlaneDistances(rasterized_point_cloud, reference_point_cloud);

    o3c::Tensor global_point_mask = reference_point_mask.LogicalAnd(global_point_mask);

    core::functional::SetMaskedToValue(distances, residual_mask, 0.0f);

    return distances;
}

open3d::core::Tensor DeformableMeshToImageFitter::ComputeHessianApproximation_BlockDiagonal(
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
) const {
    o3c::Tensor
            point_map_vectors = rasterized_point_cloud.GetPointPositions() - reference_point_cloud.GetPointPositions();
    o3c::Tensor rasterized_normals = rasterized_point_cloud.GetPointNormals();

    o3c::Tensor pixel_jacobians, node_pixel_indices_jagged, node_pixel_index_counts;
    kernel::ComputePixelVertexAnchorJacobiansAndNodeAssociations(
            pixel_jacobians, node_pixel_indices_jagged, node_pixel_index_counts,
            rasterized_vertex_position_jacobians, rasterized_vertex_normal_jacobians,
            warped_vertex_position_jacobians, warped_vertex_normal_jacobians,
            point_map_vectors, rasterized_normals, residual_mask, pixel_faces,
            warped_mesh.GetTriangleIndices(), vertex_anchors, node_count
    );

//    open3d::core::Tensor node_jacobians, node_jacobian_ranges, node_jacobian_pixel_indices;
//    kernel::ConvertPixelVertexAnchorJacobiansToNodeJacobians(
//            node_jacobians, node_jacobian_ranges, node_jacobian_pixel_indices,
//            node_pixel_vertex_jacobians, node_pixel_vertex_jacobian_counts, pixel_vertex_anchor_jacobians);
    open3d::core::Tensor hessian_approximation_blocks;
    kernel::ComputeHessianApproximationBlocks(hessian_approximation_blocks, pixel_jacobians,
                                              node_pixel_indices_jagged, node_pixel_index_counts);

    return hessian_approximation_blocks;
}


} // namespace nnrt::alignment