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
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace warp {

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void AddAnchorInfluence(Eigen::Map<Eigen::Vector3f>& warped_point, const int anchor_index, const float weight,
                               const Eigen::Vector3f& point,
                               const NDArrayIndexer& node_indexer,
                               const NDArrayIndexer& node_rotation_indexer,
                               const NDArrayIndexer& node_translation_indexer) {
	if (anchor_index != -1) {
		auto node_data = node_indexer.template GetDataPtr<float>(anchor_index);
		auto rotation_data = node_rotation_indexer.template GetDataPtr<float>(anchor_index);
		auto translation_data = node_translation_indexer.template GetDataPtr<float>(anchor_index);
		Eigen::Vector3f node(node_data);
		Eigen::Vector3f translation(translation_data);
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotation(rotation_data);
		warped_point += weight * (node + rotation * (point - node) + translation);
	}
}

template<o3c::Device::DeviceType TDeviceType>
void WarpPoints(o3c::Tensor& warped_points, const o3c::Tensor& points,
                const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                const o3c::Tensor& node_translations,
                int anchor_count, const float node_coverage) {

	const int64_t point_count = points.GetLength();
	const int64_t node_count = nodes.GetLength();

	float node_coverage_squared = node_coverage * node_coverage;

	// initialize output array
	warped_points = o3c::Tensor::Zeros({point_count, 3}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);

	//output indexer
	NDArrayIndexer warped_point_indexer(warped_points, 1);

#if defined(__CUDACC__)
	namespace launcher = o3c::kernel::cuda_launcher;
#else
	namespace launcher = o3c::kernel::cpu_launcher;
#endif
	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data);

				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];

				graph::FindAnchorsAndWeightsForPointEuclidean<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
				                                                           point, node_indexer, node_coverage_squared);

				auto warped_point_data = warped_point_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> warped_point(warped_point_data);
				for (int i_anchor_index = 0; i_anchor_index < anchor_count; i_anchor_index++) {
					int32_t anchor_index = anchor_indices[i_anchor_index];
					float weight = anchor_weights[i_anchor_index];
					AddAnchorInfluence<TDeviceType>(warped_point, anchor_index, weight, point,
					                                node_indexer, node_rotation_indexer, node_translation_indexer);
				}
			}
	);

}


template<o3c::Device::DeviceType TDeviceType>
void WarpPoints(o3c::Tensor& warped_points, const o3c::Tensor& points,
                const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                const o3c::Tensor& node_translations,
                const o3c::Tensor& anchors, const o3c::Tensor& anchor_weights) {

	const int64_t point_count = points.GetLength();
	const int64_t node_count = nodes.GetLength();
	const auto anchor_count = static_cast<int32_t>(anchors.GetShape(1));

	// initialize output array
	warped_points = o3c::Tensor::Zeros({point_count, 3}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer node_rotation_indexer(node_rotations, 1);
	NDArrayIndexer node_translation_indexer(node_translations, 1);
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer anchor_weight_indexer(anchor_weights, 1);

	//output indexer
	NDArrayIndexer warped_point_indexer(warped_points, 1);

#if defined(__CUDACC__)
	namespace launcher = o3c::kernel::cuda_launcher;
#else
	namespace launcher = o3c::kernel::cpu_launcher;
#endif
	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data);


				auto warped_point_data = warped_point_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> warped_point(warped_point_data);
				const auto* anchors = anchor_indexer.template GetDataPtr<const int32_t>(workload_idx);
				const auto* weights = anchor_weight_indexer.template GetDataPtr<const float>(workload_idx);
				for (int i_anchor_index = 0; i_anchor_index < anchor_count; i_anchor_index++) {
					const int32_t anchor_index = anchors[i_anchor_index];
					const float weight = weights[i_anchor_index];
					AddAnchorInfluence<TDeviceType>(warped_point, anchor_index, weight, point,
					                                node_indexer, node_rotation_indexer, node_translation_indexer);
				}
			}
	);

}


} // namespace warp
} // namespace kernel
} // namespace geometry
} // namespace nnrt