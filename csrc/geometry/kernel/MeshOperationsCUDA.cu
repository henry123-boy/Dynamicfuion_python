//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/21/22.
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
#include "geometry/kernel/MeshOperationsImpl.h"

namespace nnrt::geometry::kernel::mesh {

template void
ComputeTriangleNormals<open3d::core::Device::DeviceType::CUDA>(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                                                               const open3d::core::Tensor& triangle_indices);
template void
NormalizeVectors3d<open3d::core::Device::DeviceType::CUDA>(open3d::core::Tensor& vectors3d);

} // nnrt::geometry::kernel::mesh