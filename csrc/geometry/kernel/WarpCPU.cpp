//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/21.
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
#include "geometry/kernel/Warp.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "geometry/kernel/WarpImpl.h"

using namespace open3d;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace warp {

template
void WarpPoints<o3c::Device::DeviceType::CPU>(
		o3c::Tensor& warped_points, const o3c::Tensor& points,
		const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
		const o3c::Tensor& node_translations, int anchor_count, float node_coverage,
		int minimum_valid_anchor_count
);

template
void WarpPoints<o3c::Device::DeviceType::CPU>(
		o3c::Tensor& warped_points, const o3c::Tensor& points,
		const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
		const o3c::Tensor& node_translations,
		const o3c::Tensor& anchors,
		const o3c::Tensor& anchor_weights,
		int minimum_valid_anchor_count
);


} // namespace warp
} // namespace kernel
} // namespace geometry
} // namespace nnrt