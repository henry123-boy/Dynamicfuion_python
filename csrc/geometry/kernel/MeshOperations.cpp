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
#include "MeshOperations.h"
#include "core/DeviceSelection.h"

namespace o3c = open3d::core;
namespace nnrt::geometry::kernel::mesh {
void ComputeTriangleNormals(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                            const open3d::core::Tensor& triangle_indices) {
	core::ExecuteOnDevice(
			vertex_positions.GetDevice(),
			[&] { ComputeTriangleNormals<o3c::Device::DeviceType::CPU>(triangle_normals, vertex_positions, triangle_indices); },
			[&] { NNRT_IF_CUDA(ComputeTriangleNormals<o3c::Device::DeviceType::CUDA>(triangle_normals, vertex_positions, triangle_indices);); }
	);
}

} // nnrt::geometry::kernel::mesh