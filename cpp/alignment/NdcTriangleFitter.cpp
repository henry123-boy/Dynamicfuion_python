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
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "geometry/functional/PerspectiveProjection.h"
#include "geometry/functional/Comparison.h"
#include "geometry/functional/MeshFrom2dTriangle.h"
#include "core/functional/Masking.h"
#include "core/kernel/MathTypedefs.h"
#include "core/TensorRepresentationConversion.h"
#include "rendering/FlatEdgeShader.h"
#include "rendering/functional/ExtractFaceVertices.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::alignment {

inline
o3c::Tensor ComputeJacobianTJacobian(
        const o3c::Tensor& rendered_normals,
        const o3c::Tensor& rendered_points,
        const o3c::Tensor& observed_points
) {
    o3u::LogError("NotImplemented");

    return o3c::Tensor();
}

void CheckTriangleFitsNdc(
        const core::kernel::Matrix3x2f& triangle,
        const core::kernel::Matrix2f& ndc_bounds,
        const std::string& triangle_prefix
) {
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


o3tg::Image RenderMeshLines(
        const open3d::t::geometry::TriangleMesh& target_mesh,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::SizeVector& image_size
) {
    auto [face_vertices, face_mask] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(target_mesh, intrinsics, image_size);
    auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
            nnrt::rendering::RasterizeMesh(face_vertices, face_mask, image_size, 0.f, 1);
    rendering::FlatEdgeShader shader(2.0, std::array<float, 3>({1.0, 1.0, 1.0}));

    auto image = shader.ShadeMeshes(pixel_face_indices, pixel_depths, pixel_barycentric_coordinates,
                                    pixel_face_distances, o3u::nullopt);
    return image;
}


std::vector<open3d::t::geometry::Image> NdcTriangleFitter::FitTriangles(
        const open3d::core::Tensor& source_triangle,
        const open3d::core::Tensor& target_triangle,
        const open3d::core::Device& device
) {
    auto source_triangle_eigen = core::TensorToEigenMatrix<core::kernel::Matrix3x2f>(source_triangle);
    auto target_triangle_eigen = core::TensorToEigenMatrix<core::kernel::Matrix3x2f>(target_triangle);

    CheckTriangleFitsNdc(source_triangle_eigen, this->ndc_bounds, "Start");
    CheckTriangleFitsNdc(target_triangle_eigen, this->ndc_bounds, "Reference");
    const float depth = 5.0;


    std::vector<open3d::t::geometry::Image> iteration_shots;


    o3tg::TriangleMesh source_mesh =
            geometry::functional::MeshFrom2dTriangle(source_triangle, device, depth, ndc_intrinsics);
    o3tg::TriangleMesh target_mesh =
            geometry::functional::MeshFrom2dTriangle(target_triangle, device, depth, ndc_intrinsics);





    const int max_iteration_count = 10;

    o3c::Tensor current_triangle = source_triangle;

    for (int i_iteration; i_iteration < max_iteration_count; i_iteration++) {
        o3tg::TriangleMesh mesh = geometry::functional::MeshFrom2dTriangle(current_triangle, device, depth,
                                                                           ndc_intrinsics);
        auto [ndc_face_vertices, face_mask] =
                nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(mesh, intrinsics, image_size);
        auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
                nnrt::rendering::RasterizeMesh(ndc_face_vertices, face_mask, image_size, 0.f, 1);




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