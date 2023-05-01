//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/12/22.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/io/ImageIO.h>

// local includes
#include "tests/test_utils/test_utils.hpp"
#include "test_main.hpp"
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/FlatEdgeShader.h"
#include "rendering/functional/ExtractFaceVertices.h"


namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;
namespace o3u = open3d::utility;

void TestShadeTriangle_VertexColors(const o3c::Device& device) {
    o3tg::TriangleMesh mesh(device);
    mesh.SetVertexPositions(
            o3c::Tensor(
                    std::vector<float>(
                            {
                                    -1.5f, -2.2f, 5.f,
                                    -1.5f, 2.2f, 5.f,
                                    2.0f, 0.0f, 5.f,
                            }
                    ),
                    {3, 3},
                    o3c::Float32,
                    device
            )
    );
    mesh.SetTriangleIndices(
            o3c::Tensor(
                    std::vector<int64_t>({0, 1, 2}),
                    {1, 3},
                    o3c::Int64,
                    device
            )
    );
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

    auto [ndc_face_vertices, face_mask] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(mesh, intrinsics, image_size);
    auto [pixel_face_indices, pixel_depths, pixel_barycentric_coordinates, pixel_face_distances] =
            nnrt::rendering::RasterizeNdcTriangles(ndc_face_vertices, face_mask, image_size, 1.0f, 1);

    nnrt::rendering::FlatEdgeShader shader(2.0, std::array<float, 3>({1.0, 1.0, 1.0}));
    auto image = shader.ShadeMeshes(pixel_face_indices, pixel_depths, pixel_barycentric_coordinates,
                                    pixel_face_distances, o3u::nullopt);

    o3tg::Image ground_truth_image;
    o3tio::ReadImage(test::static_image_test_data_directory.ToString() + "/triangle.png", ground_truth_image);
    ground_truth_image = ground_truth_image.To(device);

    REQUIRE(image.AsTensor().AllClose(ground_truth_image.AsTensor()));
}

TEST_CASE("Test Flat Edge Shader - CPU") {
    auto device = o3c::Device("CPU:0");
    TestShadeTriangle_VertexColors(device);
}

TEST_CASE("Test Flat Edge Shader - CUDA") {
    auto device = o3c::Device("CUDA:0");
    TestShadeTriangle_VertexColors(device);
}