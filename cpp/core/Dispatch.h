//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/24/23.
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

// third-party includes
#include <open3d/core/Dtype.h>
#include <Eigen/Dense>

#ifndef DISPATCH_SIGNED_DTYPE_TO_TEMPLATE
#define DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(DTYPE, ...)                   \
    [&] {                                                        \
        if (DTYPE == open3d::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int8) {                \
            using scalar_t = int8_t;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int16) {               \
            using scalar_t = int16_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int32) {               \
            using scalar_t = int32_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int64) {               \
            using scalar_t = int64_t;                            \
            return __VA_ARGS__();                                \
        } else {                                                 \
            open3d::utility::LogError("Unsupported data type."); \
        }                                                        \
    }()
#endif

#ifndef DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE
#define DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE(SIZE, ELEMENT_TYPE, ...) \
    [&]{                                                                   \
        switch(SIZE){                                                      \
         case 1:{                                                          \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 1>;               \
            return __VA_ARGS__();                                          \
            }                                                              \
        case 2:{                                                           \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 2>;               \
            return __VA_ARGS__();                                          \
            }                                                              \
        case 3:{                                                           \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 3>;               \
            return __VA_ARGS__();                                          \
            }                                                              \
        case 4:{                                                           \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 4>;               \
            return __VA_ARGS__();                                          \
            }                                                              \
        default:                                                           \
            open3d::utility::LogError("Unsupported size, {}."              \
            " Only sizes 2-4 are supported.", SIZE);                       \
            return;                                                        \
        }                                                                  \
    }()
#endif

#ifndef DISPATCH_MATRIX_BLOCK_SIZE_TO_EIGEN_TYPE
#define DISPATCH_MATRIX_BLOCK_SIZE_TO_EIGEN_TYPE(SIZE,  ...)               \
    [&]{                                                                   \
        switch(SIZE){                                                      \
         case 3:{                                                          \
            using matrix_t = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;  \
            return __VA_ARGS__();                                          \
            }                                                              \
        case 6:{                                                           \
            using matrix_t = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>;  \
            return __VA_ARGS__();                                          \
            }                                                              \
        default:                                                           \
            open3d::utility::LogError("Unsupported size, {}."              \
            " Only sizes 2-4 are supported.", SIZE);                       \
            return;                                                        \
        }                                                                  \
    }()
#endif