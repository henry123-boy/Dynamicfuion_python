//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/12/22.
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
// stdlib
#include <algorithm>
#include <random>

// test framework
#include "test_main.hpp"
#include "tests/test_utils/geometry.h"
#include "tests/test_utils/test_utils.hpp"

// code being tested
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/functional/ExtractFaceVertices.h"
#include "core/TensorManipulationRoutines.h"

namespace o3c = open3d::core;
namespace o3g = open3d::geometry;
namespace o3tg = open3d::t::geometry;

void TestExtractFaceVertices(const o3c::Device& device) {
    auto plane = test::GenerateXyPlane(1.2615, std::make_tuple(0.f, 0.f, 1.f), 4, device);
    o3c::Tensor intrinsics(std::vector<double>{
            580., 0., 320.,
            0., 580., 240.,
            0., 0., 1.,
    }, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));

    auto [extracted_face_vertices, clipped_face_mask] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask(plane, intrinsics, {480, 640}, 0.0, 2.0);

    REQUIRE(clipped_face_mask.NonZero().GetShape(1) == 334);

    // test data generated via Python
    auto extracted_face_vertices_ground_truth = open3d::core::Tensor::Load(
            test::array_test_data_directory.ToString() + "/extracted_face_vertices.npy").To(device);
    auto clipped_face_mask_ground_truth = open3d::core::Tensor::Load(
            test::array_test_data_directory.ToString() + "/extracted_face_mask.npy").To(device);


    extracted_face_vertices.SetItem(o3c::TensorKey::IndexTensor(clipped_face_mask.LogicalNot()),
                                    nnrt::core::SingleValueTensor(0.f, device));

    REQUIRE(clipped_face_mask.AllClose(clipped_face_mask_ground_truth));
    REQUIRE(extracted_face_vertices.AllClose(extracted_face_vertices_ground_truth));
}

TEST_CASE("Test Extract Face Vertices - CPU") {
    auto device = o3c::Device("CPU:0");
    TestExtractFaceVertices(device);
}

TEST_CASE("Test Extract Face Vertices - CUDA") {
    auto device = o3c::Device("CUDA:0");
    TestExtractFaceVertices(device);
}

void TestExtractFaceVerticesMultipleMeshes(const o3c::Device& device) {
    auto plane = test::GenerateXyPlane(1.2615, std::make_tuple(0.f, 0.f, 1.f), 4, device);
    o3c::Tensor intrinsics(std::vector<double>{
            580., 0., 320.,
            0., 580., 240.,
            0., 0., 1.,
    }, {3, 3}, o3c::Float64, o3c::Device("CPU:0"));

    auto sphere = test::GenerateSphere(0.4, std::make_tuple(0.f, 0.f, 0.5f), 32, device);

    auto [extracted_face_vertices, clipped_face_mask, face_counts] =
            nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask({plane, sphere}, intrinsics, {480, 640}, 0.0, 2.0);

    REQUIRE(clipped_face_mask.NonZero().GetShape(1) == 1994);

    // test data generated via Python
    auto extracted_face_vertices_ground_truth = open3d::core::Tensor::Load(
            test::array_test_data_directory.ToString() + "/extracted_face_vertices_multiple_meshes.npy").To(device);
    auto clipped_face_mask_ground_truth = open3d::core::Tensor::Load(
            test::array_test_data_directory.ToString() + "/extracted_face_mask_multiple_meshes.npy").To(device);


    extracted_face_vertices.SetItem(o3c::TensorKey::IndexTensor(clipped_face_mask.LogicalNot()),
                                    nnrt::core::SingleValueTensor(0.f, device));

    REQUIRE(clipped_face_mask.AllClose(clipped_face_mask_ground_truth));
    REQUIRE(extracted_face_vertices.Slice(0, 0, 334).AllClose(extracted_face_vertices_ground_truth.Slice(0, 0, 334)));
}

TEST_CASE("Test Extract Face Vertices Multiple Meshes - CPU") {
    auto device = o3c::Device("CPU:0");
    TestExtractFaceVerticesMultipleMeshes(device);
}

TEST_CASE("Test Extract Face Vertices Multiple Meshes - CUDA") {
    auto device = o3c::Device("CUDA:0");
    TestExtractFaceVerticesMultipleMeshes(device);
}
