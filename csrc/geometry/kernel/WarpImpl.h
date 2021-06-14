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
#pragma once

#include <open3d/t/geometry/kernel/GeometryIndexer.h>

#include "geometry/kernel/Warp.h"
#include "geometry/kernel/Defines.h"
#include "geometry/kernel/GraphUtilitiesImpl.h"



using namespace open3d;
using namespace open3d::t::geometry::kernel;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace warp {

template<open3d::core::Device::DeviceType TDeviceType>
void WarpPoints(core::Tensor& warped_points, const core::Tensor& points,
                const core::Tensor& nodes, const core::Tensor& node_rotations,
                const core::Tensor& node_translations,
                const int anchor_count, const float node_coverage){

	const int64_t point_count = points.GetLength();
	const int64_t node_count = nodes.GetLength();

	float node_coverage_squared = node_coverage * node_coverage;

	// initialize output array
	warped_points = core::Tensor::Zeros({point_count, 3}, core::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);

	//output indexer
	NDArrayIndexer warped_point_indexer(warped_points, 1);

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

				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];
				if (!graph::FindAnchorsAndWeightsForPoint<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
				                                                       point, node_indexer, node_coverage_squared)) {
					return;
				}

				auto warped_point_data = warped_point_indexer.template GetDataPtrFromCoord<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> warped_point(warped_point_data);
				for (int i_anchor_index = 0; i_anchor_index < anchor_count; i_anchor_index++){
					int32_t anchor_index = anchor_indices[i_anchor_index];
					float weight = anchor_weights[i_anchor_index];
					if(anchor_index != -1){
						auto node_data = node_indexer.template GetDataPtrFromCoord<float>(anchor_index);
						auto rotation_data = node_rotation_indexer.template GetDataPtrFromCoord<float>(anchor_index);
						auto translation_data = node_translation_indexer.template GetDataPtrFromCoord<float>(anchor_index);
						Eigen::Vector3f node(node_data);
						Eigen::Vector3f translation(translation_data);
						Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation(rotation_data);
						warped_point += weight * (node + rotation * (point - node) + translation);
					}
				}
			}
	);

}

} // namespace warp
} // namespace kernel
} // namespace geometry
} // namespace nnrt