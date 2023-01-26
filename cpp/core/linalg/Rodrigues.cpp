//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/26/23.
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
// stdlib includes

// third-party includes

// local includes
#include "core/linalg/Rodrigues.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::linalg {

open3d::core::Tensor AxisAngleVectorsToMatricesRodrigues(const open3d::core::Tensor& vectors) {
	open3d::core::Tensor matrices;
	core::ExecuteOnDevice(
			vectors.GetDevice(),
			[&]() {
				internal::AngleAxisVectorsToMatricesRodrigues<open3d::core::Device::DeviceType::CPU>(matrices, vectors);
			},
			[&]() {
				NNRT_IF_CUDA(
						internal::AngleAxisVectorsToMatricesRodrigues<open3d::core::Device::DeviceType::CUDA>(matrices, vectors);
				);
			}
	);
	return matrices;
}

} // namespace nnrt::core::linalg
