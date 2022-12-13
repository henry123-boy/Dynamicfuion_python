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
#pragma once

// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/Image.h>

// local
#include "alignment/functional/kernel/MathTypedefs.h"

namespace nnrt::alignment {

// Exists mostly for testing purposes -- an optimization that tests various routines that are used for the
// bigger mesh-to-image fitting routine
class FlatTriangleFitter {
public:
    FlatTriangleFitter();
    std::vector<open3d::t::geometry::Image> FitFlatTriangles(
            const Matrix3x2f& start_triangle,
            const Matrix3x2f& reference_triangle,
            open3d::core::Device,
            float depth
    );

};

} // namespace nnrt::alignment
