//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/1/23.
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
#include <open3d/core/Device.h>

// local includes
#include "geometry/functional/kernel/TopologicalConversions.h"
#include "core/DeviceSelection.h"
namespace o3c = open3d::core;

namespace nnrt::geometry::functional::kernel {
void MeshToAdjacencyArray(open3d::core::Tensor& adjacency_array, const open3d::t::geometry::TriangleMesh& mesh, int max_expected_vertex_degree) {
	core::ExecuteOnDevice(
			mesh.GetDevice(),
			[&] {
				MeshToAdjacencyArray<o3c::Device::DeviceType::CPU>(adjacency_array, mesh, max_expected_vertex_degree);
			},
			[&] {
				NNRT_IF_CUDA(MeshToAdjacencyArray<o3c::Device::DeviceType::CUDA>(adjacency_array, mesh, max_expected_vertex_degree););
			}
	);
}
} // namespace nnrt::geometry::functional::kernel