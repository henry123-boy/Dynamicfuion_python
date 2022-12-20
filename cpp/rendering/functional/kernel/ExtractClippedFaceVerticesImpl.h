//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/18/22.
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
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

//local
#include "ExtractClippedFaceVertices.h"
#include "core/PlatformIndependentAtomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;


namespace nnrt::rendering::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType, bool TExtractNormals, typename TRunBeforeReturnInvalid, typename TGetOutputIndex>
void ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic(
        open3d::core::Tensor& face_vertex_positions_ndc,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_vertex_normals_camera,

        const std::vector<open3d::core::Tensor>& vertex_positions_sets_camera,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::core::Tensor>>> vertex_normals_camera,
        const std::vector<open3d::core::Tensor>& triangle_index_sets,
        const open3d::core::Tensor& face_counts,
        int64_t total_face_count,
        const open3d::core::Tensor& ndc_intrinsic_matrix,
        geometry::kernel::AxisAligned2dBoundingBox normalized_camera_space_xy_range,
        float near_clipping_distance,
        float far_clipping_distance,
        TRunBeforeReturnInvalid&& run_before_return_invalid,
        TGetOutputIndex&& get_output_face_index
) {
    o3c::Device device = vertex_positions_sets_camera[0].GetDevice();
    int64_t mesh_count = face_counts.GetLength();

    const auto* face_count_data = face_counts.GetDataPtr<int64_t>();

    const float** vertex_position_set_data_pointers;
    vertex_position_set_data_pointers = static_cast<const float**>(malloc(sizeof(float*) * mesh_count));
    for (int i_mesh = 0; i_mesh < mesh_count; i_mesh++) {
        vertex_position_set_data_pointers[i_mesh] = vertex_positions_sets_camera[i_mesh].GetDataPtr<float>();
    }
    const int64_t** triangle_index_set_data_pointers;
    triangle_index_set_data_pointers = static_cast<const int64_t**>(malloc(sizeof(int64_t*) * mesh_count));
    for (int i_mesh = 0; i_mesh < mesh_count; i_mesh++) {
        triangle_index_set_data_pointers[i_mesh] = triangle_index_sets[i_mesh].GetDataPtr<int64_t>();
    }
#ifdef __CUDACC__
    const float** vertex_position_set_data_pointers_device;
    cudaMalloc(&vertex_position_set_data_pointers_device, sizeof(float*) * mesh_count);
    cudaMemcpy(vertex_position_set_data_pointers_device, vertex_position_set_data_pointers,
               sizeof(float*) * mesh_count, cudaMemcpyHostToDevice);
    free(vertex_position_set_data_pointers);
    vertex_position_set_data_pointers = vertex_position_set_data_pointers_device;
    const int64_t** triangle_index_set_data_pointers_device;
    cudaMalloc(&triangle_index_set_data_pointers_device, sizeof(int64_t*) * mesh_count);
    cudaMemcpy(triangle_index_set_data_pointers_device, triangle_index_set_data_pointers,
               sizeof(int64_t*) * mesh_count, cudaMemcpyHostToDevice);
    free(triangle_index_set_data_pointers);
    triangle_index_set_data_pointers = triangle_index_set_data_pointers_device;
#endif

    // output & output indexers
    face_vertex_positions_ndc = open3d::core::Tensor({total_face_count, 3, 3}, o3c::Float32, device);
    o3tgk::NDArrayIndexer normalized_face_vertex_indexer(face_vertex_positions_ndc, 1);
    o3tgk::TransformIndexer perspective_transform(ndc_intrinsic_matrix,
                                                  o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0f);

    float* face_vertex_normal_ptr = nullptr;
    const float** vertex_normal_data_pointers = nullptr;
    if (TExtractNormals) {
        face_vertex_normals_camera.value().get() = open3d::core::Tensor({total_face_count, 3, 3}, o3c::Float32, device);
        face_vertex_normal_ptr = face_vertex_normals_camera.value().get().GetDataPtr<float>();
        
        
        vertex_normal_data_pointers = static_cast<const float**>(malloc(sizeof(float*) * mesh_count));
        for (int i_mesh = 0; i_mesh < mesh_count; i_mesh++) {
            vertex_normal_data_pointers[i_mesh] = vertex_normals_camera.value().get()[i_mesh].GetDataPtr<float>();
        }
        #ifdef __CUDACC__
        const float** vertex_normal_data_pointers;
        cudaMalloc(&vertex_normal_data_pointers, sizeof(float*) * mesh_count);
        cudaMemcpy(vertex_normal_data_pointers, vertex_position_set_data_pointers,
                   sizeof(float*) * mesh_count, cudaMemcpyHostToDevice);
        free(vertex_position_set_data_pointers);
        vertex_position_set_data_pointers = vertex_normal_data_pointers;
        #endif
    }


    o3c::ParallelFor(
            device, total_face_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t scene_face_idx) {
                int64_t current_face_count = 0;
                int i_mesh = -1;
                while(current_face_count <= scene_face_idx){
                    current_face_count += face_count_data[++i_mesh];
                }
                const int64_t i_face = scene_face_idx - (current_face_count - face_count_data[i_mesh]);
                const float* vertex_data = vertex_position_set_data_pointers[i_mesh];
                const int64_t* triangle_index_data = triangle_index_set_data_pointers[i_mesh];


                Eigen::Map<const Eigen::RowVector3<int64_t>> vertex_indices(triangle_index_data + i_face * 3);

                Eigen::Map<const Eigen::Vector3f> vertex0(vertex_data + vertex_indices(0) * 3);
                Eigen::Map<const Eigen::Vector3f> vertex1(vertex_data + vertex_indices(1) * 3);
                Eigen::Map<const Eigen::Vector3f> vertex2(vertex_data + vertex_indices(2) * 3);
                Eigen::Map<const Eigen::Vector3f> vertices[] = {vertex0, vertex1, vertex2};

                // clip if all vertices are too near or too far
                bool have_vertex_within_clipping_range = false;
                for (auto& vertex: vertices) {
                    have_vertex_within_clipping_range |= vertex.z() >= near_clipping_distance;
                    have_vertex_within_clipping_range |= vertex.z() <= far_clipping_distance;
                }
                if (!have_vertex_within_clipping_range) {
                    run_before_return_invalid(scene_face_idx);
                    return;
                }

                Eigen::Vector2f normalized_face_vertices_xy[3];
                bool has_inliner_vertex = false;
                for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
                    perspective_transform.Project(vertices[i_vertex].x(), vertices[i_vertex].y(),
                                                  vertices[i_vertex].z(),
                                                  &normalized_face_vertices_xy[i_vertex].x(),
                                                  &normalized_face_vertices_xy[i_vertex].y());
                    has_inliner_vertex |= normalized_camera_space_xy_range.Contains(
                            normalized_face_vertices_xy[i_vertex]);
                }
                if (!has_inliner_vertex) {
                    run_before_return_invalid(scene_face_idx);
                    return;
                }

                auto output_face_index = get_output_face_index(scene_face_idx);
                Eigen::Map<Eigen::Vector3f> normalized_face_vertex0(
                        normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index));
                Eigen::Map<Eigen::Vector3f> normalized_face_vertex1(
                        normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 3);
                Eigen::Map<Eigen::Vector3f> normalized_face_vertex2(
                        normalized_face_vertex_indexer.GetDataPtr<float>(output_face_index) + 6);
                Eigen::Map<Eigen::Vector3f> normalized_face_vertices[] = {
                        normalized_face_vertex0, normalized_face_vertex1, normalized_face_vertex2
                };
                for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
                    normalized_face_vertices[i_vertex].x() = normalized_face_vertices_xy[i_vertex].x();
                    // important: flip the y-coordinate to reflect pixel space
                    normalized_face_vertices[i_vertex].y() = normalized_face_vertices_xy[i_vertex].y();
                    normalized_face_vertices[i_vertex].z() = vertices[i_vertex].z();
                }
                if (TExtractNormals) {
                    const float* vertex_normals_data = vertex_normal_data_pointers[i_mesh];
                    for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
                        memcpy(face_vertex_normal_ptr + output_face_index * 9 + i_vertex * 3,
                               vertex_normals_data + (vertex_indices(i_vertex) * 3),
                               sizeof(float) * 3);
                    }
                }
            }
    );
#ifdef __CUDACC__
    cudaFree(vertex_position_set_data_pointers);
    cudaFree(triangle_index_set_data_pointers);
#else
    free(vertex_position_set_data_pointers);
    free(triangle_index_set_data_pointers);
#endif
    if (TExtractNormals) {
        #ifdef __CUDACC__
            cudaFree(vertex_normal_data_pointers);
        #else
            free(vertex_normal_data_pointers);
        #endif
    }
}

static
std::tuple<o3c::Tensor, int64_t> GetFaceCountHeuristics(const std::vector<open3d::core::Tensor>& triangle_index_sets) {
    int64_t total_face_count = 0;
    std::vector<int64_t> face_count_data;

    for (const auto& index_set: triangle_index_sets) {
        int64_t face_count = index_set.GetLength();
        total_face_count += face_count;
        face_count_data.push_back(face_count);
    }
    o3c::Tensor face_counts = o3c::Tensor(face_count_data, {static_cast<int64_t>(face_count_data.size())}, o3c::Int64,
                                          triangle_index_sets[0].GetDevice());
    return std::make_tuple(face_counts, total_face_count);
}

template<open3d::core::Device::DeviceType TDeviceType>
void MeshDataAndClippingMaskToNdc(
        open3d::core::Tensor& vertex_positions_normalized_camera,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_vertex_normals_camera,
        open3d::core::Tensor& clipped_face_mask,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> face_counts,
        const std::vector<open3d::core::Tensor>& vertex_positions_camera,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::core::Tensor>>> vertex_normals_camera,
        const std::vector<open3d::core::Tensor>& triangle_vertex_indices,
        const open3d::core::Tensor& ndc_intrinsics,
        geometry::kernel::AxisAligned2dBoundingBox ndc_xy_range,
        float near_clipping_distance,
        float far_clipping_distance
) {
    o3c::Device device = vertex_positions_camera[0].GetDevice();
    auto [face_counts_, total_face_count] = GetFaceCountHeuristics(triangle_vertex_indices);
    if (face_counts.has_value()) {
        face_counts.value() = face_counts_;
    }

    if (face_vertex_normals_camera.has_value() != vertex_normals_camera.has_value()) {
        utility::LogError(
                "either both or none of face_vertex_normals_camera[out] and normals_camera[in] need to have Tensor values passed in. "
                "face_vertex_normals_camera: {}; normals_camera: {}}",
                (face_vertex_normals_camera.has_value() ? " has value" : " is null"),
                (vertex_normals_camera.has_value() ? " has value" : " is null"));
    }
    bool extract_normals = vertex_normals_camera.has_value();

    clipped_face_mask = open3d::core::Tensor::Ones({total_face_count}, o3c::Bool, device);


    bool* clipped_face_mask_ptr = clipped_face_mask.template GetDataPtr<bool>();
    auto run_before_return_invalid = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
        clipped_face_mask_ptr[workload_idx] = false;
    };
    auto get_output_face_index = NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
        return workload_idx;
    };
    if (extract_normals) {
        ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic<TDeviceType, true>(
                vertex_positions_normalized_camera,
                face_vertex_normals_camera,
                vertex_positions_camera,
                vertex_normals_camera,
                triangle_vertex_indices,
                face_counts_,
                total_face_count,
                ndc_intrinsics,
                ndc_xy_range,
                near_clipping_distance,
                far_clipping_distance,
                run_before_return_invalid,
                get_output_face_index
        );
    } else {
        ExtractClippedFaceVerticesInNormalizedCameraSpace_Generic<TDeviceType, false>(
                vertex_positions_normalized_camera,
                face_vertex_normals_camera,
                vertex_positions_camera,
                vertex_normals_camera,
                triangle_vertex_indices,
                face_counts_,
                total_face_count,
                ndc_intrinsics,
                ndc_xy_range,
                near_clipping_distance,
                far_clipping_distance,
                run_before_return_invalid,
                get_output_face_index
        );
    }


}

} // namespace nnrt::rendering::functional::kernel;