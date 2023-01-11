//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/13/22.
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

#include <Eigen/Dense>

#include "core/platform_independence/Qualifiers.h"

namespace nnrt::geometry::kernel{

struct AxisAligned2dBoundingBox {
	float min_x;
	float max_x;
	float min_y;
	float max_y;

	NNRT_DEVICE_WHEN_CUDACC
	bool Contains(const Eigen::Vector2f& point) const {
		return point.y() >= min_y && point.x() >= min_x && point.y() <= max_y && point.x() <= max_x;
	}

    NNRT_DEVICE_WHEN_CUDACC
    Eigen::Matrix<float, 2, 2, Eigen::RowMajor> ToMatrix() const{
        Eigen::Matrix<float, 2, 2, Eigen::RowMajor> matrix;
        matrix << min_y, max_y, min_x, max_x;
        return matrix;
    }
};


} // namespace nnrt::rendering::kernel