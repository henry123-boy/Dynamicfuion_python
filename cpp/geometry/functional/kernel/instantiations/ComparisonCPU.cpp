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
#include "geometry/functional/kernel/Comparison.h"
#include "open3d/core/ParallelFor.h"
#include "geometry/functional/kernel/ComparisonImpl.h"

namespace nnrt::geometry::functional::kernel::comparison{
template
void ComputePointToPlaneDistances<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& distances,
		const open3d::core::Tensor& normals1,
		const open3d::core::Tensor& vertices1,
		const open3d::core::Tensor& vertices2);

} // namespace nnrt::geometry::functional::kernel::comparison