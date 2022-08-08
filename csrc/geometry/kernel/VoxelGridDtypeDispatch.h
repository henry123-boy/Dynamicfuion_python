//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/29/22.
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

namespace nnrt::geometry::kernel::voxel_grid{
#define DISPATCH_VALUE_DTYPE_TO_TEMPLATE(WEIGHT_DTYPE, COLOR_DTYPE, ...)    \
    [&] {                                                                   \
        if (WEIGHT_DTYPE == open3d::core::Float32 &&                        \
            COLOR_DTYPE == open3d::core::Float32) {                         \
            using weight_t = float;                                         \
            using color_t = float;                                          \
            return __VA_ARGS__();                                           \
        } else if (WEIGHT_DTYPE == open3d::core::UInt16 &&                  \
                   COLOR_DTYPE == open3d::core::UInt16) {                   \
            using weight_t = uint16_t;                                      \
            using color_t = uint16_t;                                       \
            return __VA_ARGS__();                                           \
        } else {                                                            \
            utility::LogError(                                              \
                    "Unsupported value data type combination. Expected "    \
                    "(float, float) or (uint16, uint16), but received ({} " \
                    "{}).",                                                 \
                    WEIGHT_DTYPE.ToString(), COLOR_DTYPE.ToString());       \
        }                                                                   \
    }()


#define DISPATCH_INPUT_DTYPE_TO_TEMPLATE(DEPTH_DTYPE, COLOR_DTYPE, ...)        \
    [&] {                                                                      \
        if (DEPTH_DTYPE == open3d::core::Float32 &&                            \
            COLOR_DTYPE == open3d::core::Float32) {                            \
            using input_depth_t = float;                                       \
            using input_color_t = float;                                       \
            return __VA_ARGS__();                                              \
        } else if (DEPTH_DTYPE == open3d::core::UInt16 &&                      \
                   COLOR_DTYPE == open3d::core::UInt8) {                       \
            using input_depth_t = uint16_t;                                    \
            using input_color_t = uint8_t;                                     \
            return __VA_ARGS__();                                              \
        } else {                                                               \
            utility::LogError(                                                 \
                    "Unsupported input data type combination. Expected "       \
                    "(float, float) or (uint16, uint8), but received ({} {})", \
                    DEPTH_DTYPE.ToString(), COLOR_DTYPE.ToString());           \
        }                                                                      \
    }()

} // nnrt::geometry::kernel::voxel_grid