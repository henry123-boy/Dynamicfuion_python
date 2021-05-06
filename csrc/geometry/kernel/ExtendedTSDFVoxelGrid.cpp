//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
//  Copyright (c) 2021 Gregory Kramida
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
#include <open3d/core/Tensor.h>
#include <open3d/utility/Console.h>

#include "geometry/kernel/ExtendedTSDFVoxelGrid.h"

using namespace open3d;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {

void ExtractVoxelCenters(const open3d::core::Tensor& block_indices, const open3d::core::Tensor& nb_block_indices,
                         const open3d::core::Tensor& nb_block_masks, const open3d::core::Tensor& block_keys,
                         const open3d::core::Tensor& block_values, open3d::core::Tensor& voxel_centers,
                         int64_t block_resolution, float voxel_size) {
	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();
	if (device_type == core::Device::DeviceType::CPU) {
		ExtractVoxelCentersCPU(block_indices,
		                       nb_block_indices, nb_block_masks, block_keys,
		                       block_values, voxel_centers,
		                       block_resolution, voxel_size);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		ExtractVoxelCentersCUDA(block_indices,
		                        nb_block_indices, nb_block_masks, block_keys,
		                        block_values, voxel_centers,
		                        block_resolution, voxel_size);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}

}


void ExtractValuesInExtent(
		int64_t min_voxel_x, int64_t min_voxel_y, int64_t min_voxel_z,
		int64_t max_voxel_x, int64_t max_voxel_y, int64_t max_voxel_z,
		const core::Tensor& block_indices, const core::Tensor& nb_block_indices, const core::Tensor& nb_block_masks,
		const core::Tensor& block_keys, const core::Tensor& block_values, core::Tensor& voxel_values, int64_t block_resolution) {

	core::Device device = block_keys.GetDevice();

	core::Device::DeviceType device_type = device.GetType();

	if (device_type == core::Device::DeviceType::CPU) {
		ExtractValuesInExtentCPU(min_voxel_x, min_voxel_y, min_voxel_z,
		                         max_voxel_x, max_voxel_y, max_voxel_z,
		                         block_indices,
		                         nb_block_indices, nb_block_masks, block_keys,
		                         block_values, voxel_values,
		                         block_resolution);
	} else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
		ExtractValuesInExtentCUDA(min_voxel_x, min_voxel_y, min_voxel_z,
		                          max_voxel_x, max_voxel_y, max_voxel_z,
		                          block_indices,
		                          nb_block_indices, nb_block_masks, block_keys,
		                          block_values, voxel_values,
		                          block_resolution);
#else
		utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
	} else {
		utility::LogError("Unimplemented device");
	}
}

} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt