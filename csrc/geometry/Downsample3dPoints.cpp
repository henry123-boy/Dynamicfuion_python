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
#include "geometry/Downsample3dPoints.h"
#include <open3d/core/TensorCheck.h>
#include "geometry/kernel/PointDownsampling.h"
#include <open3d/core/hashmap/HashSet.h>

namespace o3c = open3d::core;

namespace nnrt::geometry {

open3d::core::Tensor
GridDownsample3dPoints(const open3d::core::Tensor& original_points, float grid_cell_size, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, { original_points.GetLength(), 3 });


	o3c::Tensor downsampled_points;
	kernel::downsampling::GridDownsamplePoints(downsampled_points, original_points, grid_cell_size, hash_backend);
	return downsampled_points;
}

open3d::core::Tensor
RadiusDownsample3dPoints(const open3d::core::Tensor& original_points, float radius, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, { original_points.GetLength(), 3 });


	o3c::Tensor downsampled_points;
	kernel::downsampling::RadiusDownsamplePoints(downsampled_points, original_points, radius, hash_backend);
	return downsampled_points;
}

} // namespace nnrt::geometry
