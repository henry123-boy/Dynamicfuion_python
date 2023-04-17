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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/Image.h>
#include <open3d/t/io/ImageIO.h>

// local includes
#include "tests/test_utils/test_main.hpp"
#include "tests/test_utils/test_utils.hpp"
#include "alignment/NdcTriangleFitter.h"
#include "core/kernel/MathTypedefs.h"
#include "core/TensorRepresentationConversion.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace o3tio = open3d::t::io;

void TestTriangleFitter(const o3c::Device& device) {
    nnrt::alignment::NdcTriangleFitter fitter({480, 640});

    nnrt::core::kernel::Matrix3x2f source_triangle, target_triangle;
    source_triangle << -1.185f, -0.655f,
            0.215f, 1.15f,
            0.395f, -0.305f;

    target_triangle << -0.75f, -1.1f,
            -0.75f, 1.1f,
            1.f, 0.f;

    std::cout << "Start triangle: " << std::endl << source_triangle << std::endl;
    std::cout << "Reference triangle: " << std::endl << target_triangle << std::endl;

    std::vector<o3tg::Image> diagnostic_images =
            fitter.FitTriangles(nnrt::core::EigenMatrixToTensor(source_triangle, device),
                                nnrt::core::EigenMatrixToTensor(target_triangle, device), device);

    o3tio::WriteImage(test::generated_image_test_data_directory.ToString() + "/fit_triangle_000_reference.png",
                      diagnostic_images[0]);

}

TEST_CASE("Test Triangle Fitter - CPU") {
    auto device = o3c::Device("CPU:0");
    TestTriangleFitter(device);
}