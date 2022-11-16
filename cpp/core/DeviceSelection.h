//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/5/21.
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
#pragma once

#include <open3d/core/Tensor.h>

namespace nnrt::core {

#ifdef BUILD_CUDA_MODULE
#define NNRT_IF_CUDA(...) __VA_ARGS__ static_assert(true)
#else
#define NNRT_IF_CUDA(...) static_assert(true)
#endif

template<typename FExecuteOnCPU, typename FExecuteOnCUDA>
void ExecuteOnDevice(open3d::core::Device device,
                     FExecuteOnCPU&& execute_on_cpu,
                     FExecuteOnCUDA&& execute_on_cuda) {
	open3d::core::Device::DeviceType device_type = device.GetType();

	switch (device_type) {
		case open3d::core::Device::DeviceType::CPU:
			execute_on_cpu();
			break;
		case open3d::core::Device::DeviceType::CUDA:
#ifdef BUILD_CUDA_MODULE
			execute_on_cuda();
#else
			open3d::utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
			break;
		default:
			open3d::utility::LogError("Unimplemented device");
			break;
	}
}

//TODO: this function is a bit too convoluted. Replace its usages with direct usages of ExecuteOnDevice
template<typename TEntity, typename FExecuteOnCPU, typename FExecuteOnCUDA>
void InferDeviceFromEntityAndExecute(const TEntity& guiding_entity,
                                     FExecuteOnCPU&& execute_on_cpu,
                                     FExecuteOnCUDA&& execute_on_cuda) {
	open3d::core::Device device = guiding_entity.GetDevice();
	ExecuteOnDevice(device, execute_on_cpu, execute_on_cuda);
}

} // nnrt::core