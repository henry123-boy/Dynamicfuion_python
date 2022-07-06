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
#include <open3d/core/hashmap/HashSet.h>

namespace o3c = open3d::core;

namespace nnrt::geometry {

// open3d::core::Tensor Downsample3DPointsByRadius(const open3d::core::Tensor& original_points, float radius) {
// 	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
// 	o3c::AssertTensorShape(original_points, {original_points.GetLength(), 3});
// 	o3c::Tensor downsampled_points;
// 	kernel::downsampling::DownsamplePointsByRadius(downsampled_points, original_points, radius);
// 	return downsampled_points;
// }

open3d::core::Tensor GridDownsample3DPoints_PlainBinArray(const open3d::core::Tensor& original_points, float grid_cell_size) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, { original_points.GetLength(), 3 });
	o3c::Tensor downsampled_points;
	kernel::downsampling::GridDownsamplePoints_PlainBinArray(downsampled_points, original_points, grid_cell_size);
	return downsampled_points;
}

open3d::core::Tensor GridDownsample3DPoints_BinHash(const open3d::core::Tensor& original_points, float grid_cell_size,
                                                    const open3d::core::HashBackendType& backend, int max_points_per_bin) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, { original_points.GetLength(), 3 });
	o3c::Device device = original_points.GetDevice();

	o3c::Tensor point_bins_float = original_points / grid_cell_size;
	o3c::Tensor point_bins_integer = point_bins_float.Floor().To(o3c::Int32);


	o3c::HashMap point_bin_coord_map(point_bins_integer.GetLength(), o3c::Int32, {3}, o3c::Int32, {1}, device, backend);

	// activate entries in the hash map that correspond to the
	o3c::Tensor buffer_indices, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, buffer_indices, success_mask);
	int64_t bin_count = success_mask.To(o3c::Int64).Sum({0}).ToFlatVector<int64_t>()[0];

	auto bin_indices = o3c::Tensor::Arange(0, bin_count, 1, o3c::Int32, device);
	point_bin_coord_map.GetValueTensor().Slice(0, 0, bin_count) = bin_indices;

	std::tie(buffer_indices, success_mask) = point_bin_coord_map.Find(point_bins_integer);

	o3c::Tensor bin_point_counts = o3c::Tensor::Zeros({bin_count}, o3c::Int32, device);
	o3c::Tensor binned_point_indices({bin_count, max_points_per_bin}, o3c::Int32, device);

	o3c::Tensor downsampled_points;
	kernel::downsampling::GridDownsamplePoints_BinHash(downsampled_points, point_bin_coord_map.GetValueTensor(), bin_point_counts,
	                                                    binned_point_indices, buffer_indices);
	return downsampled_points;
}

} // namespace nnrt::geometry
