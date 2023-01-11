//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/10/23.
//  Copyright (c) 2023 Gregory Kramida
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
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>


// local includes
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/linalg/KroneckerTensorProduct.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/ParallelFor.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

#define MAX_PIXELS_PER_NODE 1500



namespace nnrt::alignment::kernel {

template<open3d::core::Device::DeviceType TDevice>
void ComputeHessianApproximation_BlockDiagonal(
        open3d::core::Tensor& pixel_jacobians,
        open3d::core::Tensor& node_jacobians,
        open3d::core::Tensor& node_jacobian_lists,
        const open3d::core::Tensor& rasterized_vertex_position_jacobians,
        const open3d::core::Tensor& rasterized_vertex_normal_jacobians,
        const open3d::core::Tensor& warped_vertex_position_jacobians,
        const open3d::core::Tensor& warped_vertex_normal_jacobians,
        const open3d::core::Tensor& point_map_vectors,
        const open3d::core::Tensor& rasterized_normals,
        const open3d::core::Tensor& residual_mask,
        const open3d::core::Tensor& pixel_faces,
        const open3d::core::Tensor& face_vertices,
        const open3d::core::Tensor& vertex_anchors,
        int64_t node_count
) {
    // === dimension checks ===
    int64_t image_height = rasterized_vertex_position_jacobians.GetShape(0);
    int64_t image_width = rasterized_vertex_position_jacobians.GetShape(1);
    int64_t pixel_count = image_width * image_height;

    o3c::Device device = residual_mask.GetDevice();

    o3c::AssertTensorShape(rasterized_vertex_position_jacobians, { image_height, image_width, 3, 9 });
    o3c::AssertTensorDevice(rasterized_vertex_position_jacobians, device);
    o3c::AssertTensorDtype(rasterized_vertex_position_jacobians, o3c::Float32);

    o3c::AssertTensorShape(rasterized_vertex_normal_jacobians, { image_height, image_width, 3, 10 });
    o3c::AssertTensorDevice(rasterized_vertex_normal_jacobians, device);
    o3c::AssertTensorDtype(rasterized_vertex_normal_jacobians, o3c::Float32);

    int64_t vertex_count = warped_vertex_position_jacobians.GetShape(0);
    int64_t anchor_count_per_vertex = warped_vertex_position_jacobians.GetShape(1);

    o3c::AssertTensorShape(warped_vertex_position_jacobians, { vertex_count, anchor_count_per_vertex, 4 });
    o3c::AssertTensorDevice(warped_vertex_position_jacobians, device);
    o3c::AssertTensorDtype(warped_vertex_position_jacobians, o3c::Float32);

    o3c::AssertTensorShape(warped_vertex_normal_jacobians, { vertex_count, anchor_count_per_vertex, 3 });
    o3c::AssertTensorDevice(warped_vertex_normal_jacobians, device);
    o3c::AssertTensorDtype(warped_vertex_normal_jacobians, o3c::Float32);

    o3c::AssertTensorShape(point_map_vectors, { pixel_count, 3 });
    o3c::AssertTensorDevice(point_map_vectors, device);
    o3c::AssertTensorDtype(point_map_vectors, o3c::Float32);

    o3c::AssertTensorShape(rasterized_normals, { pixel_count, 3 });
    o3c::AssertTensorDevice(rasterized_normals, device);
    o3c::AssertTensorDtype(rasterized_normals, o3c::Float32);

    o3c::AssertTensorShape(residual_mask, { pixel_count });
    o3c::AssertTensorDtype(residual_mask, o3c::Bool);

    int64_t faces_per_pixel = pixel_faces.GetShape(2);
    o3c::AssertTensorShape(pixel_faces, { image_height, image_width, faces_per_pixel });
    o3c::AssertTensorDevice(pixel_faces, device);
    o3c::AssertTensorDtype(pixel_faces, o3c::Int64);

    o3c::AssertTensorShape(face_vertices, { utility::nullopt, 3 });
    o3c::AssertTensorDevice(face_vertices, device);
    o3c::AssertTensorDtype(face_vertices, o3c::Int64);

    o3c::AssertTensorShape(vertex_anchors, { vertex_count, anchor_count_per_vertex });
    o3c::AssertTensorDevice(vertex_anchors, device);
    o3c::AssertTensorDtype(vertex_anchors, o3c::Int32);

    // === initialize output matrices ===
    node_jacobian_lists = o3c::Tensor({node_count, MAX_PIXELS_PER_NODE}, o3c::Int32);
    node_jacobian_lists.Fill(-1);
    auto node_pixel_list_data = node_jacobian_lists.GetDataPtr<int32_t>();

    core::AtomicCounterArray node_jacobian_counts(node_count);


    // pixel count x 3 vertices per face x 4 anchors per vertex x 6 values per node (3 rotation angles, 3 translation coordinates)
    pixel_jacobians = o3c::Tensor({pixel_count, 3, 4, 6}, o3c::Float32);
    auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();

    // === get access to raw input data
    auto residual_mask_data = residual_mask.GetDataPtr<bool>();
    auto point_map_vector_data = point_map_vectors.GetDataPtr<float>();
    auto rasterized_normal_data = rasterized_normals.GetDataPtr<float>();

    auto rasterized_vertex_position_jacobian_data = rasterized_vertex_position_jacobians.GetDataPtr<float>();
    auto rasterized_vertex_normal_jacobian_data = rasterized_vertex_normal_jacobians.GetDataPtr<float>();

    auto warped_vertex_position_jacobian_data = warped_vertex_position_jacobians.GetDataPtr<float>();
    auto warped_vertex_normal_jacobian_data = warped_vertex_normal_jacobians.GetDataPtr<float>();

    auto pixel_face_data = pixel_faces.template GetDataPtr<int64_t>();
    auto triangle_index_data = face_vertices.GetDataPtr<int64_t>();
    auto vertex_anchor_data = vertex_anchors.GetDataPtr<int32_t>();

    // === loop over all pixels & compute
    //__DEBUG
    //o3c::ParallelFor(
    core::ParallelForMutable(
            device, pixel_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t pixel_index) mutable {
                if (!residual_mask_data[pixel_index]) {
                    return;
                }
                auto v_image = static_cast<int>(pixel_index / image_width);
                auto u_image = static_cast<int>(pixel_index % image_width);
                Eigen::Map<const Eigen::RowVector3f> pixel_point_map_vector(point_map_vector_data + pixel_index);
                Eigen::Map<const Eigen::RowVector3f> pixel_rasterized_normal(rasterized_normal_data + pixel_index);

                Eigen::Map<const core::kernel::Matrix3x9f> pixel_vertex_position_jacobian
                        (rasterized_vertex_position_jacobian_data + pixel_index * (3 * 9));
                Eigen::Map<const core::kernel::Matrix3x9f> pixel_rasterized_normal_jacobian
                        (rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10));
                Eigen::Map<const Eigen::RowVector3f> pixel_barycentric_coordinates
                        (rasterized_vertex_normal_jacobian_data + pixel_index * (3 * 10) + (3 * 9));
                // r stands for "residuals"
                // wl and nl stand for rasterized vertex positions and normals, respectively
                // V and N stand for warped vertex positions and normals, respectively
                // [1 x 3] * [3 x 9] = [1 x 9]
                auto dr_dwl_x_dwl_dV = pixel_rasterized_normal * pixel_vertex_position_jacobian;
                // [1 x 3] * [3 x 9] = [1 x 9]
                auto dr_dnl_x_dnl_dV = pixel_point_map_vector * pixel_rasterized_normal_jacobian;
                // [1 x 6] * [6 x 9] = [1 x 9]
                auto dr_dV = dr_dwl_x_dwl_dV + dr_dnl_x_dnl_dV;
                // dr_dN = dr_dl * dl_dV + dr_dn * dl_dN = 0 + dr_dn   * dl_dN
                //                                             [1 x 3] * [3 x 9] = [1 x 9]
                auto dr_dN =
                        pixel_point_map_vector *
                        Eigen::kroneckerProduct(pixel_barycentric_coordinates, core::kernel::Matrix3f::Identity());

                auto i_face = pixel_face_data[(v_image * image_width * faces_per_pixel) + (u_image * faces_per_pixel)];
                Eigen::Map<const Eigen::RowVector3<int64_t>> vertex_indices(triangle_index_data + i_face * 3);
                int pixel_jacobian_address = pixel_index * (3 * 4 * 6);
                auto current_pixel_jacobian_data = pixel_jacobian_data + pixel_jacobian_address;
                //TODO: try to optimize in CUDA-only version using 3 x anchors_per_vertex blocks and block "shared"
                // memory... Or, at the very least, try increasing thread count to pixel_count x 3 x anchors_per_vertex
                // instead of just pixel_count.
                for (int i_face_vertex = 0; i_face_vertex < 3; i_face_vertex++) {
                    int64_t i_vertex = vertex_indices(i_face_vertex);
                    int vertex_jacobian_address = i_face_vertex * (4 * 6);
                    auto pixel_vertex_jacobian_data = current_pixel_jacobian_data + vertex_jacobian_address;
                    for (int i_anchor = 0; i_anchor < anchor_count_per_vertex; i_anchor++) {
                        auto i_node = vertex_anchor_data[i_vertex * anchor_count_per_vertex + i_anchor];
                        if (i_node == -1) {
                            // -1 is sentinel value for anchors
                            continue;
                        }
                        int anchor_jacobian_address = i_anchor * 6;
                        int i_node_jacobian = node_jacobian_counts.FetchAdd(i_node, 1);
                        node_pixel_list_data[i_node * MAX_PIXELS_PER_NODE + i_node_jacobian] =
                                pixel_jacobian_address + vertex_jacobian_address + anchor_jacobian_address;

                        // [3x3] warped vertex position Jacobian w.r.t. node rotation
                        const Eigen::SkewSymmetricMatrix3<float> dv_drotation(
                                Eigen::Map<const Eigen::Vector3f>(
                                        warped_vertex_position_jacobian_data +
                                        (i_vertex * anchor_count_per_vertex * 4) +
                                        (i_anchor * 4)
                                )
                        );
                        // used to compute warped vertex position Jacobian w.r.t. node translation, weight * I_3x3
                        float stored_node_weight =
                                warped_vertex_position_jacobian_data[
                                        (i_vertex * anchor_count_per_vertex * 4) +
                                        (i_anchor * 4) + 3];
                        // [3x3] warped vertex normal Jacobian w.r.t. node rotation
                        const Eigen::SkewSymmetricMatrix3<float> dn_drotation(
                                Eigen::Map<const Eigen::Vector3f>(
                                        warped_vertex_normal_jacobian_data +
                                        (i_vertex * anchor_count_per_vertex * 3) +
                                        (i_anchor * 3)
                                )
                        );
                        // [1x3]
                        auto dr_dv = dr_dV.block<1,3>(0, i_vertex * 3);
                        // [1x3]
                        auto dr_dn = dr_dN.block<1,3>(0, i_vertex * 3);

                        Eigen::Map<Eigen::RowVector3<float>>
                                pixel_vertex_anchor_rotation_jacobian(pixel_vertex_jacobian_data + anchor_jacobian_address);
                        Eigen::Map<Eigen::RowVector3<float>>
                                pixel_vertex_anchor_translation_jacobian(pixel_vertex_jacobian_data + anchor_jacobian_address + 3);

                        // [1x3] = ([1x3] * [3x3]) + ([1x3] * [3x3])
                        pixel_vertex_anchor_rotation_jacobian = (dr_dv * dv_drotation) + (dr_dn * dn_drotation);
                        pixel_vertex_anchor_translation_jacobian = dr_dv * (Eigen::Matrix3f::Identity() * stored_node_weight);
                    }
                }
            }
    );


}

} // namespace nnrt::alignment::kernel