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
#include "core/platform_independence/Atomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

#define MAX_JACOBIANS_PER_NODE 1500


namespace nnrt::alignment::kernel {

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Swap(TScalar* a, TScalar* b) {
    TScalar tmp = *a;
    *a = *b;
    *b = tmp;
}

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Heapify(TScalar* array, int length, int root) {
    int largest = root;
    int l = 2 * root + 1;
    int r = 2 * root + 2;

    if (l < length && array[l] > array[largest]) {
        largest = l;
    }
    if (r < length && array[r] > array[largest]) {
        largest = r;
    }
    if (largest != root) {
        Swap<TScalar>(&array[root], &array[largest]);
        Heapify<TScalar>(array, length, largest);
    }
}

template<typename TScalar>
NNRT_DEVICE_WHEN_CUDACC void HeapSort(TScalar* array, int length) {
    for (int i = length / 2 - 1; i >= 0; i--) Heapify(array, length, i);

    for (int i = length - 1; i > 0; i--) {
        Swap<TScalar>(&array[0], &array[i]);
        Heapify<TScalar>(array, i, 0);
    }
}

template<open3d::core::Device::DeviceType TDeviceType>
void ComputePixelVertexAnchorJacobiansAndNodeAssociations(
        open3d::core::Tensor& pixel_vertex_anchor_jacobians,
        open3d::core::Tensor& node_pixel_vertex_jacobians,
        open3d::core::Tensor& node_pixel_vertex_jacobian_counts,
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
    // === dimension, type, and device tensor checks ===
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
    node_pixel_vertex_jacobians = o3c::Tensor({node_count, MAX_JACOBIANS_PER_NODE}, o3c::Int32);
    node_pixel_vertex_jacobians.Fill(-1);
    auto node_pixel_vertex_jacobian_data = node_pixel_vertex_jacobians.GetDataPtr<int32_t>();

    core::AtomicCounterArray<TDeviceType> node_pixel_vertex_jacobian_counters(node_count);

    // pixel count x 3 vertices per face x 4 anchors per vertex x 6 values per node (3 rotation angles, 3 translation coordinates)
    pixel_vertex_anchor_jacobians = o3c::Tensor({pixel_count, 3, 4, 6}, o3c::Float32);
    auto pixel_vertex_anchor_jacobian_data = pixel_vertex_anchor_jacobians.GetDataPtr<float>();

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
    o3c::ParallelFor(
            device, pixel_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t pixel_index) {
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
                auto current_pixel_jacobian_data = pixel_vertex_anchor_jacobian_data + pixel_jacobian_address;
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
                        int i_node_jacobian = node_pixel_vertex_jacobian_counters.FetchAdd(i_node, 1);
                        node_pixel_vertex_jacobian_data[i_node * MAX_JACOBIANS_PER_NODE + i_node_jacobian] =
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
                        auto dr_dv = dr_dV.block<1, 3>(0, i_vertex * 3);
                        // [1x3]
                        auto dr_dn = dr_dN.block<1, 3>(0, i_vertex * 3);

                        Eigen::Map<Eigen::RowVector3<float>>
                                pixel_vertex_anchor_rotation_jacobian
                                (pixel_vertex_jacobian_data + anchor_jacobian_address);
                        Eigen::Map<Eigen::RowVector3<float>>
                                pixel_vertex_anchor_translation_jacobian
                                (pixel_vertex_jacobian_data + anchor_jacobian_address + 3);

                        // [1x3] = ([1x3] * [3x3]) + ([1x3] * [3x3])
                        pixel_vertex_anchor_rotation_jacobian = (dr_dv * dv_drotation) + (dr_dn * dn_drotation);
                        pixel_vertex_anchor_translation_jacobian =
                                dr_dv * (Eigen::Matrix3f::Identity() * stored_node_weight);
                    }
                }
            }
    );
    node_pixel_vertex_jacobian_counts = node_pixel_vertex_jacobian_counters.AsTensor(true);
}

template <open3d::core::Device::DeviceType TDevice>
void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
        open3d::core::Tensor& node_jacobians,
        open3d::core::Tensor& node_jacobian_ranges,
        open3d::core::Tensor& node_jacobian_pixel_indices,
        open3d::core::Tensor& node_pixel_vertex_jacobians, // in: unsorted; out: sorted
        const open3d::core::Tensor& node_pixel_vertex_jacobian_counts,
        const open3d::core::Tensor& pixel_vertex_anchor_jacobians
){
    // === dimension, type, and device tensor checks ===
    int64_t node_count = node_pixel_vertex_jacobian_counts.GetShape(0);

    o3c::AssertTensorShape(node_pixel_vertex_jacobian_counts, {node_count});
    o3c::Device device = node_pixel_vertex_jacobian_counts.GetDevice();
    o3c::AssertTensorDtype(node_pixel_vertex_jacobian_counts, o3c::Int32);

    o3c::AssertTensorShape(node_pixel_vertex_jacobians, {node_count, MAX_JACOBIANS_PER_NODE});
    o3c::AssertTensorDtype(node_pixel_vertex_jacobians, o3c::Int32);
    o3c::AssertTensorDevice(node_pixel_vertex_jacobians, device);

    o3c::AssertTensorShape(pixel_vertex_anchor_jacobians, {utility::nullopt, 3, 4, 6});
    o3c::AssertTensorDtype(pixel_vertex_anchor_jacobians, o3c::Float32);
    o3c::AssertTensorDevice(pixel_vertex_anchor_jacobians, device);

    // === get access to input arrays ===
    auto node_pixel_vertex_jacobian_data = node_pixel_vertex_jacobians.GetDataPtr<int32_t>();
    auto node_pixel_vertex_jacobian_count_data = node_pixel_vertex_jacobian_counts.GetDataPtr<int32_t>();
    auto pixel_vertex_anchor_jacobian_data = pixel_vertex_anchor_jacobians.GetDataPtr<float>();

    // === set up atomic counter ===
    NNRT_DECLARE_ATOMIC(int64_t, total_jacobian_count);
    NNRT_INITIALIZE_ATOMIC(int64_t, total_jacobian_count, 0L);

    // === set up output tensor to store ranges ===
    node_jacobian_ranges = o3c::Tensor ({node_count, 2}, o3c::Int64, device);
    auto node_jacobian_range_data = node_jacobian_ranges.GetDataPtr<int64_t>();
    // === loop over all nodes and sort all entries by jacobian address
    o3c::ParallelFor(
            device, node_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
                const int* node_list_start = node_pixel_vertex_jacobian_data + (node_index * MAX_JACOBIANS_PER_NODE);
                int node_list_length = node_pixel_vertex_jacobian_count_data[node_index];
                // sort the anchor jacobian addresses
                HeapSort(node_list_start, node_list_length);

                int node_jacobian_count = 0;
                int previous_pixel_index = -1;
                // count up the sorted addresses sans those with repeating pixel index
                for(int i_jacobian = 0; i_jacobian < node_list_length; i_jacobian++){
                    // -1 is the sentinel value
                    int jacobian_address = node_list_start[i_jacobian];
                    int pixel_index = jacobian_address / (3 * 4 * 6);
                    if (pixel_index != previous_pixel_index) {
                        node_jacobian_count++;
                    }
                }
                node_jacobian_range_data[node_index * 2 + 1] = node_jacobian_count;
                NNRT_ATOMIC_ADD(total_jacobian_count, static_cast<int64_t>(node_jacobian_count));
            }
    );

    node_jacobians = o3c::Tensor({NNRT_GET_ATOMIC_VALUE_HOST(total_jacobian_count), 6}, o3c::Float32, device);
    auto node_jacobian_data = node_jacobians.GetDataPtr<float>();

    node_jacobian_pixel_indices = o3c::Tensor({node_jacobians.GetShape(0)}, o3c::Int32, device);
    auto node_jacobian_pixel_index_data = node_jacobian_pixel_indices.GetDataPtr<int32_t>();

    NNRT_CLEAN_UP_ATOMIC(total_jacobian_count);

    o3c::ParallelFor(
            device, node_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
                // figure out where to start filling in the jacobians for the current node
                int node_jacobian_start_index = 0;
                for (int i_node = 0; i_node < node_index-1; i_node++){
                    node_jacobian_start_index += node_jacobian_range_data[i_node * 2 + 1];
                }
                node_jacobian_range_data[node_index * 2] = node_jacobian_start_index;
                int node_jacobian_count = node_jacobian_range_data[node_index * 2 + 1];

                // source data to get anchor jacobians from
                int* node_list_start = node_pixel_vertex_jacobian_data + (node_index * MAX_JACOBIANS_PER_NODE);
                int node_list_length = node_pixel_vertex_jacobian_count_data[node_index];

                // loop over source data and fill in node jacobians & corresponding pixel indices
                int i_node_jacobian = 0;
                int previous_pixel_address = -1;
                for (int i_anchor_jacobian = 0; i_anchor_jacobian < node_list_length; i_anchor_jacobian++){
                    int pixel_jacobian_address = node_list_start[i_anchor_jacobian];
                    int pixel_index = pixel_jacobian_address / (3 * 4 * 6);
                    // source anchor jacobian
                    Eigen::Map<const Eigen::Matrix<float, 1, 6>> pixel_anchor_jacobian(pixel_vertex_anchor_jacobian_data + pixel_jacobian_address);
                    Eigen::Map<const Eigen::Matrix<float, 1, 6>> node_jacobian(node_jacobian_data + i_node_jacobian * 6);

                    if (pixel_index != previous_pixel_address) {
                        node_jacobian_pixel_index_data[i_node_jacobian] = pixel_index;
                        node_jacobian = pixel_anchor_jacobian;
                        i_node_jacobian++;
                    } else {
                        // if the jacobian corresponds to the same pixel (but different vertex), add it to the
                        // aggregate jacobian for the same node.
                        node_jacobian += pixel_anchor_jacobian;
                    }
                }
            }
    );
}

} // namespace nnrt::alignment::kernel