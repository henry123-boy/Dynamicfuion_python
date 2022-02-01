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
#include "geometry/kernel/WarpUtilities.h"


using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt::geometry::kernel::warp {

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


template<o3c::Device::DeviceType TDeviceType, typename TFindAnchorsFunction>
void WarpPoints_OnlineAnchors_Generic(o3c::Tensor& warped_points, const o3c::Tensor& points,
                                      const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                                      const o3c::Tensor& node_translations,
                                      int anchor_count, const float node_coverage, TFindAnchorsFunction&& find_anchors) {
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

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data);

				int32_t anchor_indices[MAX_ANCHOR_COUNT];
				float anchor_weights[MAX_ANCHOR_COUNT];

				if (!find_anchors(anchor_indices, anchor_weights, node_count,
				                  point, node_indexer, node_coverage_squared)) {
					return;
				}

				auto warped_point_data = warped_point_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> warped_point(warped_point_data);
				warp::BlendWarp(warped_point, anchor_indices, anchor_weights, anchor_count, node_indexer, node_rotation_indexer,
				                node_translation_indexer, point);
			}
	);
}

template<o3c::Device::DeviceType TDeviceType>
void WarpPoints(o3c::Tensor& warped_points, const o3c::Tensor& points,
                const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                const o3c::Tensor& node_translations,
                int anchor_count, const float node_coverage, int minimum_valid_anchor_count) {
	WarpPoints_OnlineAnchors_Generic<TDeviceType>(
			warped_points, points, nodes, node_rotations, node_translations, anchor_count, node_coverage,
			[=] NNRT_DEVICE_WHEN_CUDACC(
					int32_t* anchor_indices, float* anchor_weights, const int node_count,
					const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
					const float node_coverage_squared) {
				return warp::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
				                                                                           minimum_valid_anchor_count, node_count,
				                                                                           point, node_indexer, node_coverage_squared);
			}
	);
}

template<o3c::Device::DeviceType TDeviceType>
void WarpPoints(o3c::Tensor& warped_points, const o3c::Tensor& points,
                const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                const o3c::Tensor& node_translations,
                int anchor_count, const float node_coverage) {
	WarpPoints_OnlineAnchors_Generic<TDeviceType>(
			warped_points, points, nodes, node_rotations, node_translations, anchor_count, node_coverage,
			[=] NNRT_DEVICE_WHEN_CUDACC(
					int32_t* anchor_indices, float* anchor_weights, const int node_count,
					const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
					const float node_coverage_squared) {
				warp::FindAnchorsAndWeightsForPointEuclidean<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
				                                                          node_count, point, node_indexer, node_coverage_squared);
				return true;
			}
	);
}

template<o3c::Device::DeviceType TDeviceType, typename TBlendWarp>
void WarpPoints_PrecomputedAnchors_Generic(o3c::Tensor& warped_points, const o3c::Tensor& points,
                                           const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                                           const o3c::Tensor& node_translations,
                                           const o3c::Tensor& anchors, const o3c::Tensor& anchor_weights,
                                           TBlendWarp&& blend_warp) {

	const int64_t point_count = points.GetLength();
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

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f source_point(point_data);


				auto warped_point_data = warped_point_indexer.template GetDataPtr<float>(workload_idx);
				Eigen::Map<Eigen::Vector3f> warped_point(warped_point_data);
				const auto* node_anchors = anchor_indexer.template GetDataPtr<const int32_t>(workload_idx);
				const auto* node_anchor_weights = anchor_weight_indexer.template GetDataPtr<const float>(workload_idx);
				blend_warp(warped_point, node_anchors, node_anchor_weights,
				           anchor_count, node_indexer, node_rotation_indexer,
				           node_translation_indexer, source_point);
			}
	);

}

template<o3c::Device::DeviceType TDeviceType>
void WarpPoints(o3c::Tensor& warped_points, const o3c::Tensor& points,
                const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                const o3c::Tensor& node_translations,
                const o3c::Tensor& anchors, const o3c::Tensor& anchor_weights,
                int minimum_valid_anchor_count) {
	WarpPoints_PrecomputedAnchors_Generic<TDeviceType>(
			warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights,
			[=]NNRT_DEVICE_WHEN_CUDACC(Eigen::Map<Eigen::Vector3f>& warped_point,
			                           const int32_t* anchor_indices, const float* anchor_weights, const int anchor_count,
			                           const NDArrayIndexer& node_indexer, const NDArrayIndexer& node_rotation_indexer,
			                           const NDArrayIndexer& node_translation_indexer, const Eigen::Vector3f& source_point) {
				warp::BlendWarp_ValidAnchorCountThreshold(
						warped_point, anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, node_indexer, node_rotation_indexer,
						node_translation_indexer, source_point
				);
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void WarpPoints(open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
                const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations,
                const open3d::core::Tensor& node_translations,
                const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights) {
	WarpPoints_PrecomputedAnchors_Generic(
			warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights,
			warp::BlendWarp<TDeviceType, Eigen::Map<Eigen::Vector3f>>
	);
}


} // namespace nnrt::geometry::kernel::warp