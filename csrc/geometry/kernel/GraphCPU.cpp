//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/8/21.
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
#include "geometry/kernel/Graph.h"
#include "open3d/core/ParallelFor.h"
#include "core/CPU/DeviceHeapCPU.h"
#include "geometry/kernel/GraphImpl.h"


namespace nnrt::geometry::kernel::graph {

template
void ComputeAnchorsAndWeightsEuclidean<open3d::core::Device::DeviceType::CPU, true>(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights,
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const int anchor_count,
		const int minimum_valid_anchor_count,
		const float node_coverage
);

template
void ComputeAnchorsAndWeightsEuclidean<open3d::core::Device::DeviceType::CPU, false>(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights,
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const int anchor_count,
		const int minimum_valid_anchor_count,
		const float node_coverage
);


template
void ComputeAnchorsAndWeightsShortestPath<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights,
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges,
		const int anchor_count,
		const float node_coverage
);

} // namespace nnrt::geometry::kernel::graph