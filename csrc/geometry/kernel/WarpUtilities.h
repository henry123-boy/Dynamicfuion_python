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

#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <Eigen/Dense>

#include "geometry/kernel/Defines.h"
#include "core/kernel/KnnUtilities.h"
#include "core/DeviceHeap.h"
#include "geometry/DualQuaternion.h"


namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt::geometry::kernel::warp {

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void NormalizeAnchorWeights(float* anchor_weights, const float weight_sum, const int anchor_count, const int valid_anchor_count) {
	if (weight_sum > 0.0f) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] /= weight_sum;
		}
	} else if (valid_anchor_count > 0) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] = 1.0f / static_cast<float>(valid_anchor_count);
		}
	}
}

template<o3c::Device::DeviceType TDeviceType, bool TDistancesAreSquared>
NNRT_DEVICE_WHEN_CUDACC
inline void
ComputeAnchorWeights(float* anchor_weights, float& weight_sum, const float* distances, const int anchor_count, const float node_coverage_squared) {
	weight_sum = 0.0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float square_distance = distances[i_anchor];
		if (!TDistancesAreSquared) {
			square_distance = square_distance * square_distance;
		}
		float weight = expf(-square_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
	}
}

template<o3c::Device::DeviceType TDeviceType, bool TDistancesAreSquared>
NNRT_DEVICE_WHEN_CUDACC
inline void
ComputeAnchorWeights_Threshold(int* anchor_indices, float* anchor_weights, float& weight_sum, int& valid_anchor_count, const float* distances,
                               const int anchor_count,
                               const float node_coverage_squared) {
	weight_sum = 0.0;
	valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float square_distance = distances[i_anchor];
		if (!TDistancesAreSquared) {
			square_distance = square_distance * square_distance;
		}
		// note: equivalent to distance > 2 * node_coverage, avoids sqrtf
		if (square_distance > 4 * node_coverage_squared) {
			anchor_indices[i_anchor] = -1;
			continue;
		}
		float weight = expf(-square_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPointEuclidean(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                       const int node_count, const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
                                       const float node_coverage_squared) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKNNAnchorsBruteForce<TDeviceType>(anchor_indices, squared_distances, anchor_count, node_count, point, node_indexer);
	float weight_sum;
	ComputeAnchorWeights<TDeviceType, true>(anchor_weights, weight_sum, squared_distances, anchor_count, node_coverage_squared);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);

}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPointEuclidean_Threshold(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                                 const int minimum_valid_anchor_count, const int node_count, const Eigen::Vector3f& point,
                                                 const NDArrayIndexer& node_indexer, const float node_coverage_squared) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKNNAnchorsBruteForce<TDeviceType>(anchor_indices, squared_distances, anchor_count, node_count, point, node_indexer);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold<TDeviceType, true>(anchor_indices, anchor_weights, weight_sum, valid_anchor_count, squared_distances, anchor_count,
	                                                  node_coverage_squared);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, valid_anchor_count);
	return true;
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPointShortestPath(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                          const int node_count, const Eigen::Vector3f& point,
                                          const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer, const float node_coverage_squared) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold shortest path distances
	core::kernel::knn::FindShortestPathKNNAnchors<TDeviceType>(anchor_indices, distances, anchor_count, node_count, point, node_indexer, edge_indexer);
	float weight_sum = 0.0;
	ComputeAnchorWeights<TDeviceType, false>(anchor_weights, weight_sum, distances, anchor_count, node_coverage_squared);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPointShortestPath_Threshold(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                                    const int minimum_valid_anchor_count, const int node_count, const Eigen::Vector3f& point,
                                                    const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer,
                                                    const float node_coverage_squared) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold shortest path distances
	core::kernel::knn::FindShortestPathKNNAnchors<TDeviceType>(anchor_indices, distances, anchor_count, node_count, point, node_indexer, edge_indexer);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold<TDeviceType, false>(anchor_indices, anchor_weights, weight_sum, valid_anchor_count, distances, anchor_count,
	                                                   node_coverage_squared);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
	return true;
}


template<typename TPoint>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarpDualQuaternions(
		TPoint& warped_point,
		const int32_t* anchor_indices, const float* anchor_weights, const int anchor_count,
		const NDArrayIndexer& node_transform_indexer, const Eigen::Vector3f& source_point
) {
	// *** linearly blend the anchor nodes' dual quaternions ***
	float coefficients[8];
	for (float& coefficient: coefficients) {
		coefficient = 0.0f;
	}
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		int anchor_node_index = anchor_indices[i_anchor];
		if (anchor_node_index != -1) {
			float anchor_weight = anchor_weights[i_anchor];
			auto node_transform = node_transform_indexer.GetDataPtr<float>(anchor_node_index);
			for (int i_coefficient = 0; i_coefficient < 8; i_coefficient++) {
				coefficients[i_coefficient] += anchor_weight * node_transform[i_coefficient];
			}
		}
	}
	Eigen::DualQuaternion<float> voxel_transformation(
			Eigen::Quaternion<float>(coefficients[0], coefficients[1], coefficients[2], coefficients[3]),
			Eigen::Quaternion<float>(coefficients[4], coefficients[5], coefficients[6], coefficients[7])
	);

	voxel_transformation.normalize();
	warped_point = voxel_transformation.transformPoint(source_point);
}

template<typename TPoint>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarpMatrices(
		TPoint& warped_point,
		const int32_t* anchor_indices, const float* anchor_weights, const int anchor_count,
		const NDArrayIndexer& node_indexer, const NDArrayIndexer& node_rotation_indexer,
		const NDArrayIndexer& node_translation_indexer, const Eigen::Vector3f& source_point
) {
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		int anchor_node_index = anchor_indices[i_anchor];
		if (anchor_node_index != -1) {
			float anchor_weight = anchor_weights[i_anchor];
			auto node_rotation_data = node_rotation_indexer.GetDataPtr<float>(anchor_node_index);
			auto node_translation_data = node_translation_indexer.GetDataPtr<float>(anchor_node_index);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> node_rotation(node_rotation_data);
			Eigen::Vector3f node_translation(node_translation_data);
			auto node_pointer = node_indexer.GetDataPtr<float>(anchor_node_index);
			Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
			warped_point += anchor_weight * (node + node_rotation * (source_point - node) + node_translation);
		}
	}
}

template<typename TPoint>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarpMatrices_ValidAnchorCountThreshold(
		TPoint& warped_point,
		const int32_t* anchor_indices, const float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count,
		const NDArrayIndexer& node_indexer, const NDArrayIndexer& node_rotation_indexer,
		const NDArrayIndexer& node_translation_indexer, const Eigen::Vector3f& source_point
) {
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		valid_anchor_count += static_cast<int>(anchor_indices[i_anchor] != -1);
	}
	if (valid_anchor_count >= minimum_valid_anchor_count) {
		BlendWarpMatrices(warped_point, anchor_indices, anchor_weights, anchor_count, node_indexer,
		                  node_rotation_indexer, node_translation_indexer, source_point);
	}
}


} // namespace nnrt::geometry::kernel::warp