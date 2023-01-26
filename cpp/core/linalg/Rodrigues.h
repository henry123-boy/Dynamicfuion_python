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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/core/Tensor.h>

// local includes
namespace nnrt::core::linalg {
open3d::core::Tensor AxisAngleVectorsToMatricesRodrigues(const open3d::core::Tensor& vectors);
namespace internal {
	template<open3d::core::Device::DeviceType TDeviceType>
	void AngleAxisVectorsToMatricesRodrigues(open3d::core::Tensor& matrices, const open3d::core::Tensor& vectors);
}
}//namespace nnrt::core::linalg


