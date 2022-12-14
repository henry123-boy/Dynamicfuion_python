//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/6/22.
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
// stdlib includes

// third-party includes
#include <Eigen/Dense>

// local includes
typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
typedef Eigen::Matrix<float, 2, 2, Eigen::RowMajor> Matrix2f;

typedef Eigen::Matrix<float, 3, 2, Eigen::RowMajor> Matrix3x2f;
typedef Eigen::Matrix<float, 2, 3, Eigen::RowMajor> Matrix2x3f;
typedef Eigen::Matrix<float, 3, 9, Eigen::RowMajor> Matrix3x9f;