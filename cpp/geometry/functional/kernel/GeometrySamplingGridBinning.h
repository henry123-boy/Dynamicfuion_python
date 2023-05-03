//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/3/23.
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
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/HashMap.h>
#include <Eigen/Dense>

// local includes

//*** === header for kernel usage only. === ***

namespace o3c = open3d::core;
namespace nnrt::geometry::functional::kernel::sampling {

template<open3d::core::Device::DeviceType TDeviceType, typename TBinType>
std::tuple<open3d::core::Tensor, int64_t, o3c::HashMap> GridBinPoints(
		const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend, const Eigen::Vector3f& grid_offset = Eigen::Vector3f::Zero()) {
	auto device = original_points.GetDevice();

	o3c::Tensor grid_offset_tensor(std::vector<float>{grid_offset.x(), grid_offset.y(), grid_offset.z()}, {1, 3}, o3c::Float32, device);

	o3c::Tensor point_bins_float = original_points / grid_cell_size + grid_offset_tensor;
	o3c::Tensor point_bins_integer = point_bins_float.Floor().To(o3c::Int32);

	o3c::HashMap point_bin_coord_map(point_bins_integer.GetLength(), o3c::Int32, {3}, o3c::UInt8, {sizeof(TBinType)}, device,
	                                 hash_backend);

	o3c::Tensor bin_indices_tensor, success_mask;
	point_bin_coord_map.Activate(point_bins_integer, bin_indices_tensor, success_mask);

	o3c::Tensor bins_integer = point_bins_integer.GetItem(o3c::TensorKey::IndexTensor(success_mask));
	auto bin_count = bins_integer.GetLength();

	std::tie(bin_indices_tensor, success_mask) = point_bin_coord_map.Find(point_bins_integer);
	return std::make_tuple(bin_indices_tensor, bin_count, point_bin_coord_map);
}

} // namespace nnrt::geometry::functional::kernel::sampling