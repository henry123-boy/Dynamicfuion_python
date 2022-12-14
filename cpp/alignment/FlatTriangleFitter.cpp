//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/13/22.
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
// 3rd-party
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>

// local
#include "alignment/FlatTriangleFitter.h"
#include "rendering/RasterizeMesh.h"
#include "rendering/functional/InterpolateFaceAttributes.h"
#include "geometry/functional/Unproject3dPoints.h"
#include "geometry/functional/Comparison.h"
#include "core/functional/Masking.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

inline
o3tg::TriangleMesh MeshFrom2dTriangle(const Matrix3x2f& triangle, const o3c::Device& device, float depth) {
    o3tg::TriangleMesh mesh(device);
    std::vector<float> triangle_vertices = {
            triangle(0, 0), triangle(0, 1), depth,
            triangle(1, 0), triangle(1, 1), depth,
            triangle(2, 0), triangle(2, 1), depth
    };
    std::vector<float> triangle_normals = {
            0.f, 0.f, -1.f,
            0.f, 0.f, -1.f,
            0.f, 0.f, -1.f
    };

    mesh.SetVertexPositions(o3c::Tensor(triangle_vertices, {3, 3}, o3c::Float32, device));
    mesh.SetTriangleIndices(o3c::Tensor(std::vector<int64_t>({0, 1, 2}), {1, 3}, o3c::Int64, device));
    mesh.SetVertexNormals(o3c::Tensor(triangle_normals, {3, 3}, o3c::Float32, device));
    return mesh;
}

inline
o3c::Tensor ComputeJacobianTJacobian(const o3c::Tensor& rendered_normals, const o3c::Tensor& rendered_points, const o3c::Tensor& observed_points, ){
    o3u::LogError("NotImplemented");

    return o3c::Tensor();
}


std::vector<open3d::t::geometry::Image> FlatTriangleFitter::FitFlatTriangles(
        const Matrix3x2f& start_triangle,
        const Matrix3x2f& reference_triangle,
        open3d::core::Device device,
        float depth
) {
    std::vector<open3d::t::geometry::Image> iteration_shots;

    o3tg::TriangleMesh reference_mesh = MeshFrom2dTriangle(reference_triangle, device, depth);
    o3tg::TriangleMesh start_mesh = MeshFrom2dTriangle(start_triangle, device, depth);

    o3c::Tensor intrinsics(
            std::vector<double>(
                    {
                            500.0, 0.0, 320.0,
                            0.0, 500.0, 240.0,
                            0.0, 0.0, 1.0
                    }), {3, 3}, o3c::Float64, o3c::Device("CPU:0"
            )
    );
    o3c::SizeVector image_size = {480, 640};

    auto [reference_ndc_face_vertices, reference_face_mask] =
            nnrt::rendering::MeshFaceVerticesAndClipMaskToNdc(reference_mesh, intrinsics, image_size);
    auto [reference_pixel_face_indices, reference_pixel_depths, reference_pixel_barycentric_coordinates, reference_pixel_face_distances] =
            nnrt::rendering::RasterizeMesh(reference_ndc_face_vertices, reference_face_mask, image_size, 0.f, 1);

    const float background_factor = 2.0f;
    const float background_depth = background_factor * depth;

    o3c::Tensor rendered_point_mask =
            nnrt::core::functional::ReplaceValue(reference_pixel_depths, -1.f, background_depth);

    iteration_shots.emplace_back(((reference_pixel_depths / (background_depth)) * 255).To(o3c::UInt8));

    o3c::Tensor reference_3d_points, dummy;
    geometry::functional::Unproject3dPointsWithoutDepthFiltering(
            reference_3d_points, dummy, reference_pixel_depths, intrinsics,
            o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.f, background_depth, true
    );
    o3tg::PointCloud reference_cloud(reference_3d_points.Reshape({-1, 3}));

    const int max_iteration_count = 10;

    Matrix3x2f current_triangle = start_triangle;

    for (int i_iteration; i_iteration < max_iteration_count; i_iteration++) {
        o3tg::TriangleMesh mesh = MeshFrom2dTriangle(current_triangle, device, depth);
        auto [ndc_face_vertices, face_mask] =
                nnrt::rendering::MeshFaceVerticesAndClipMaskToNdc(mesh, intrinsics, image_size);
        auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
                nnrt::rendering::RasterizeMesh(ndc_face_vertices, face_mask, image_size, 0.f, 1);
        o3c::Tensor point_mask =
                nnrt::core::functional::ReplaceValue(pixel_depths, -1.f, background_depth);
        o3c::Tensor rendered_points;
        geometry::functional::Unproject3dPointsWithoutDepthFiltering(
                rendered_points, dummy, pixel_depths, intrinsics,
                o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.f, background_depth, true
        );
        o3c::Tensor rendered_normals =
                nnrt::rendering::functional::InterpolateFaceAttributes(pixel_face_indices,
                                                                       pixel_barycentric_coordinates,
                                                                       mesh.GetVertexNormals());
        o3tg::PointCloud rendered_cloud(rendered_points.Reshape({-1, 3}));
        rendered_cloud.SetPointNormals(rendered_normals.Reshape({-1, 3}));

        o3c::Tensor point_to_plane_distances =
                geometry::functional::ComputePointToPlaneDistances(rendered_cloud, reference_cloud);

        o3c::Tensor jacobianT_jacobian = ComputeJacobianTJacobian();

    }


    return iteration_shots;
}

FlatTriangleFitter::FlatTriangleFitter() {
}

} // namespace nnrt::alignment