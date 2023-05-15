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
#include "GeometrySampling.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::functional::kernel::sampling {


void GridDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend
) {
	core::ExecuteOnDevice(
			original_points.GetDevice(),
			[&] {
				GridDownsamplePoints<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, grid_cell_size, hash_backend);
			},
			[&] {
				NNRT_IF_CUDA(
						GridDownsamplePoints<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, grid_cell_size,
						                                                    hash_backend);
				);
			}
	);
}

void FastRadiusDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float min_distance,
		const open3d::core::HashBackendType& hash_backend
) {
	core::ExecuteOnDevice(
			original_points.GetDevice(),
			[&] {
				FastRadiusDownsamplePoints<o3c::Device::DeviceType::CPU>(downsampled_points, original_points, min_distance, hash_backend);
			},
			[&] {
				NNRT_IF_CUDA(
						FastRadiusDownsamplePoints<o3c::Device::DeviceType::CUDA>(downsampled_points, original_points, min_distance,
						                                                          hash_backend);
				);
			}
	);
}

void GridMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend_type
) {
	core::ExecuteOnDevice(
			points.GetDevice(),
			[&] {
				GridMedianSubsample3dPoints<o3c::Device::DeviceType::CPU>(sample, points, grid_cell_size, hash_backend_type);
			},
			[&] {
				NNRT_IF_CUDA(
						GridMedianSubsample3dPoints<o3c::Device::DeviceType::CUDA>(sample, points, grid_cell_size, hash_backend_type);
				);
			}
	);
}

void RadiusMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float radius,
		const open3d::core::HashBackendType& hash_backend_type
) {
	core::ExecuteOnDevice(
			points.GetDevice(),
			[&] {
				RadiusMedianSubsample3dPoints<o3c::Device::DeviceType::CPU>(sample, points, radius, hash_backend_type);
			},
			[&] {
				NNRT_IF_CUDA(
						RadiusMedianSubsample3dPoints<o3c::Device::DeviceType::CUDA>(sample, points, radius, hash_backend_type);
				);
			}
	);
}


void RadiusSubsampleGraph(
		open3d::core::Tensor& resampled_vertices,
		open3d::core::Tensor& resampled_edges,
		const open3d::core::Tensor& vertices,
		const open3d::core::Tensor& edges,
		float radius
) {
	core::ExecuteOnDevice(
			vertices.GetDevice(),
			[&] {
				RadiusSubsampleGraph<o3c::Device::DeviceType::CPU>(resampled_vertices, resampled_edges, vertices, edges, radius);
			},
			[&] {
				NNRT_IF_CUDA(
						RadiusSubsampleGraph<o3c::Device::DeviceType::CUDA>(resampled_vertices, resampled_edges, vertices, edges, radius);
				);
			}
	);
}

void GridMedianSubsample3dPointsWithBinInfo(
		open3d::core::Tensor& sample,
		open3d::core::Tensor& other_bin_point_indices,
		const open3d::core::Tensor& points,
		float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend_type
) {
	core::ExecuteOnDevice(
			points.GetDevice(),
			[&] {
				GridMedianSubsample3dPointsWithBinInfo<o3c::Device::DeviceType::CPU>(sample, other_bin_point_indices, points,
				                                                                     grid_cell_size,
				                                                                     hash_backend_type);
			},
			[&] {
				NNRT_IF_CUDA(
						GridMedianSubsample3dPointsWithBinInfo<o3c::Device::DeviceType::CUDA>(sample, other_bin_point_indices,
						                                                                      points, grid_cell_size,
						                                                                      hash_backend_type);
				);
			}
	);
}

} // nnrt::geometry::functional::kernel::sampling