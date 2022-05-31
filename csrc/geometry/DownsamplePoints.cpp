//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/25/22.
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
#include "geometry/DownsamplePoints.h"
#include <open3d/core/TensorCheck.h>
#include "geometry/kernel/PointDownsampling.h"

namespace o3c = open3d::core;

namespace nnrt::geometry {

open3d::core::Tensor Downsample3DPointsByRadius(const open3d::core::Tensor& original_points, float radius) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, {original_points.GetLength(), 3});
	o3c::Tensor downsampled_points;
	kernel::downsampling::DownsamplePointsByRadius(downsampled_points, original_points, radius);
	return downsampled_points;
}

open3d::core::Tensor GridDownsample3DPoints(const open3d::core::Tensor& original_points, float grid_cell_size) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, {original_points.GetLength(), 3});
	o3c::Tensor downsampled_points;
	kernel::downsampling::GridDownsamplePoints(downsampled_points, original_points, grid_cell_size);
	return downsampled_points;
}

} // namespace nnrt::geometry
