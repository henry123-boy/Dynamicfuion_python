//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/4/21.
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
#include "geometry/kernel/Comparison.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::kernel::comparison {

void ComputePointToPlaneDistances(open3d::core::Tensor& distances,
                                  const open3d::core::Tensor& normals1,
                                  const open3d::core::Tensor& vertices1,
                                  const open3d::core::Tensor& vertices2) {
	core::InferDeviceFromEntityAndExecute(
			normals1,
			[&] { ComputePointToPlaneDistances<o3c::Device::DeviceType::CPU>(distances, normals1, vertices1, vertices2); },
			[&] { NNRT_IF_CUDA(ComputePointToPlaneDistances<o3c::Device::DeviceType::CUDA>(distances, normals1, vertices1, vertices2);); }
	);
}

} // nnrt::geometry::kernel::comparison
