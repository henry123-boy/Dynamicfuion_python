//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/18/22.
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

#include "core/PlatformIndependence.h"
#include <open3d/core/Device.h>
#include <Eigen/Dense>

namespace nnrt::geometry::kernel {

template<open3d::core::Device::DeviceType DeviceType>
NNRT_DEVICE_WHEN_CUDACC inline int
Ravel3dGridToLinearIndex(const Eigen::Vector3i& grid_coordinate,
                         const Eigen::Vector3i& grid_extent) {
	return grid_extent.z() * grid_extent.y() * grid_coordinate.z() +
	       grid_extent.y() * grid_coordinate.y() +
	       grid_coordinate.x();
}

template<open3d::core::Device::DeviceType DeviceType>
NNRT_DEVICE_WHEN_CUDACC inline Eigen::Vector3i
UnravelLinearIndexTo3dGrid(int linear_index,
                           const Eigen::Vector3i& grid_extent) {
	Eigen::Vector3i cell_coordinate;
	const int yz_layer_count = grid_extent.z() * grid_extent.y();
	cell_coordinate.z() = linear_index / yz_layer_count;
	cell_coordinate.y() = (linear_index % yz_layer_count) / grid_extent.y();
	cell_coordinate.x() = (linear_index % yz_layer_count) % grid_extent.y();
	return cell_coordinate;
}
} // nnrt::geometry::kernel