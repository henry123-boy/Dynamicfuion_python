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
#pragma once

#include <cmath>
#include <Eigen/Dense>

#include <open3d/core/Tensor.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

#include "utility/PlatformIndependence.h"
#include "geometry/kernel/GraphUtilitiesImpl.h"

using namespace open3d;
using namespace open3d::t::geometry::kernel;


namespace nnrt {
namespace geometry {
namespace kernel {
namespace graph {

template<open3d::core::Device::DeviceType TDeviceType>
void ComputeAnchorsAndWeightsEuclidean
		(open3d::core::Tensor& anchors,
		 open3d::core::Tensor& weights,
		 const open3d::core::Tensor& points,
		 const open3d::core::Tensor& nodes,
		 const int anchor_count,
		 const float node_coverage) {

	float node_coverage_squared = node_coverage * node_coverage;

	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = core::Tensor::Ones({point_count, anchor_count}, core::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = core::Tensor({point_count, anchor_count}, core::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

#if defined(__CUDACC__)
	core::CUDACachedMemoryManager::ReleaseCache();
#endif
#if defined(__CUDACC__)
	core::kernel::CUDALauncher launcher;
#else
	core::kernel::CPULauncher launcher;
#endif
	launcher.LaunchGeneralKernel(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx){
				auto point_data = point_indexer.GetDataPtrFromCoord<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtrFromCoord<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtrFromCoord<float>(workload_idx);
				if (!graph::FindAnchorsAndWeightsForPoint<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
				                                                       point, node_indexer, node_coverage_squared)) {
					return;
				}
				// endregion
			}
	);
}

} // namespace graph
} // namespace kernel
} // namespace geometry
} // namespace nnrt