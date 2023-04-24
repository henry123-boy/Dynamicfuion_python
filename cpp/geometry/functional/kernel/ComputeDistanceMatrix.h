//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/24/23.
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



namespace nnrt::geometry::functional::kernel {

void ComputeDistanceMatrix(open3d::core::Tensor& i_matrix_entry, const open3d::core::Tensor& point_set1, const open3d::core::Tensor& point_set2);

template <open3d::core::Device::DeviceType TDeviceType>
void ComputeDistanceMatrix(open3d::core::Tensor& i_matrix_entry, const open3d::core::Tensor& point_set1, const open3d::core::Tensor& point_set2);

} // namespace nnrt::geometry::functional::kernel