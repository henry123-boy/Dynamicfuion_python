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
#include <open3d/t/geometry/kernel/PointCloud.h>
#include <open3d/t/geometry/Utility.h>

// local
#include "alignment/NdcTriangleFitter.h"
#include "rendering/RasterizeMesh.h"
#include "rendering/functional/InterpolateFaceAttributes.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/Comparison.h"
#include "core/functional/Masking.h"
#include "rendering/kernel/CoordinateSystemConversions.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

inline o3tg::TriangleMesh MeshFrom2dTriangle(
        const Matrix3x2f& triangle_ndc,
        const o3c::Device& device,
        float depth,
        const o3c::Tensor& ndc_intrinsics
) {
    o3tg::CheckIntrinsicTensor(ndc_intrinsics);
    o3tg::TriangleMesh mesh(device);
    std::vector<float> projected_vertex_data = {
            triangle_ndc(0, 0), triangle_ndc(0, 1),
            triangle_ndc(1, 0), triangle_ndc(1, 1),
            triangle_ndc(2, 0), triangle_ndc(2, 1),
    };
    o3c::Tensor projected_vertices(projected_vertex_data, {3, 2}, o3c::Float32, device);
    o3c::Tensor projected_vertex_depth(std::vector<float>({depth}), {1}, o3c::Float32, device);

    o3c::Tensor vertex_positions =
            geometry::functional::UnprojectProjectedPoints(projected_vertices, projected_vertex_depth, ndc_intrinsics);

    std::vector<float> triangle_normals = {
            0.f, 0.f, -1.f,
            0.f, 0.f, -1.f,
            0.f, 0.f, -1.f
    };

    mesh.SetVertexPositions(vertex_positions);
    mesh.SetTriangleIndices(o3c::Tensor(std::vector<int64_t>({0, 1, 2}), {1, 3}, o3c::Int64, device));
    mesh.SetVertexNormals(o3c::Tensor(triangle_normals, {3, 3}, o3c::Float32, device));
    return mesh;
}

inline
o3c::Tensor ComputeJacobianTJacobian(
        const o3c::Tensor& rendered_normals,
        const o3c::Tensor& rendered_points,
        const o3c::Tensor& observed_points
) {
    o3u::LogError("NotImplemented");

    return o3c::Tensor();
}

void CheckTriangleFitsNdc(const Matrix3x2f& triangle, const Matrix2f& ndc_bounds, const std::string& triangle_prefix) {
    for (int i_vertex; i_vertex < 3; i_vertex++) {
        if (triangle(i_vertex, 0) < ndc_bounds(1, 0) || triangle(i_vertex, 0) > ndc_bounds(1, 1) ||
            triangle(i_vertex, 1) < ndc_bounds(0, 0) || triangle(i_vertex, 1) > ndc_bounds(0, 1)) {
            o3u::LogError("{} triangle vertex x:{} y:{} out-of-bounds x_min:{} x_max:{} y_min:{} y_max{}",
                          triangle_prefix,
                          triangle(i_vertex, 0), triangle(i_vertex, 1),
                          ndc_bounds(1, 0), ndc_bounds(1, 1), ndc_bounds(0, 0), ndc_bounds(0, 1));
        }
    }
}


std::vector<open3d::t::geometry::Image> NdcTriangleFitter::FitTriangles(
        const Matrix3x2f& start_triangle,
        const Matrix3x2f& reference_triangle,
        const open3d::core::Device& device
) {
    CheckTriangleFitsNdc(start_triangle, this->ndc_bounds, "Start");
    CheckTriangleFitsNdc(reference_triangle, this->ndc_bounds, "Reference");
    const float depth = 5.0;


    std::vector<open3d::t::geometry::Image> iteration_shots;

    o3tg::TriangleMesh reference_mesh = MeshFrom2dTriangle(reference_triangle, device, depth, ndc_intrinsics);
    o3tg::TriangleMesh start_mesh = MeshFrom2dTriangle(start_triangle, device, depth, ndc_intrinsics);


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
    geometry::functional::UnprojectDepthImageWithoutFiltering(
            reference_3d_points, dummy, reference_pixel_depths, intrinsics,
            o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.f, background_depth, true
    );
    o3tg::PointCloud reference_cloud(reference_3d_points.Reshape({-1, 3}));

    const int max_iteration_count = 10;

    Matrix3x2f current_triangle = start_triangle;

    for (int i_iteration; i_iteration < max_iteration_count; i_iteration++) {
        o3tg::TriangleMesh mesh = MeshFrom2dTriangle(current_triangle, device, depth, ndc_intrinsics);
        auto [ndc_face_vertices, face_mask] =
                nnrt::rendering::MeshFaceVerticesAndClipMaskToNdc(mesh, intrinsics, image_size);
        auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
                nnrt::rendering::RasterizeMesh(ndc_face_vertices, face_mask, image_size, 0.f, 1);
        o3c::Tensor point_mask =
                nnrt::core::functional::ReplaceValue(pixel_depths, -1.f, background_depth);
        o3c::Tensor rendered_points;
        geometry::functional::UnprojectDepthImageWithoutFiltering(
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

//        o3c::Tensor jacobianT_jacobian = ComputeJacobianTJacobian();

    }


    return iteration_shots;
}

NdcTriangleFitter::NdcTriangleFitter(const open3d::core::SizeVector& image_size_pixels) {
    if (image_size_pixels.size() > 2) {
        o3u::LogError("Image size is expected to be a vector of size 2, i.e. <height, width>, got size: {}",
                      image_size_pixels.size());
    }
    this->image_size = image_size_pixels;

    int64_t height = image_size[0];
    int64_t width = image_size[1];
    const double smaller_dimension_fov_radians = 0.8726; // 50 degrees
    double fx, fy;
    if (height > width) {
        fx = static_cast<double>(width / 2) / tan(smaller_dimension_fov_radians / 2);
        fy = fx;
    } else {
        fy = static_cast<double>(height / 2) / tan(smaller_dimension_fov_radians / 2);
        fx = fy;
    }

    intrinsics = o3c::Tensor(
            std::vector<double>(
                    {
                            fx, 0.0, static_cast<double>(image_size[1] / 2),
                            0.0, fy, static_cast<double>(image_size[0] / 2),
                            0.0, 0.0, 1.0
                    }
            ), {3, 3}, o3c::Float64, o3c::Device("CPU:0")
    );

    auto [ndc_intrinsics_, ndc_range] = rendering::kernel::ImageSpaceIntrinsicsToNdc(intrinsics, image_size);
    this->ndc_intrinsics = ndc_intrinsics_;
    this->ndc_bounds = ndc_range.ToMatrix();
}

} // namespace nnrt::alignment