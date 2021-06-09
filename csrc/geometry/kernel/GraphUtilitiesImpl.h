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

#include "geometry/kernel/Defines.h"

using namespace open3d;
using namespace open3d::t::geometry::kernel;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace graph {

template<core::Device::DeviceType TDeviceType>
NNRT_CPU_OR_CUDA_DEVICE
inline void FindKNNAnchorsBruteForce(int32_t* anchor_indices, float* squared_distances, const int anchor_count,
                                     const int node_count, const Eigen::Vector3f& voxel_global_metric,
                                     const NDArrayIndexer& node_indexer) {
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		squared_distances[i_anchor] = INFINITY;
	}
	int max_at_index = 0;
	float max_squared_distance = INFINITY;
	for (int i_node = 0; i_node < node_count; i_node++) {
		auto node_pointer = node_indexer.GetDataPtrFromCoord<float>(i_node);
		Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
		float squared_distance = (node - voxel_global_metric).squaredNorm();

		if (squared_distance < max_squared_distance) {
			squared_distances[max_at_index] = squared_distance;
			anchor_indices[max_at_index] = i_node;

			//update the maximum distance within current anchor nodes
			max_at_index = 0;
			max_squared_distance = squared_distances[max_at_index];
			for (int i_anchor = 1; i_anchor < anchor_count; i_anchor++) {
				if (squared_distances[i_anchor] > max_squared_distance) {
					max_at_index = i_anchor;
					max_squared_distance = squared_distances[i_anchor];
				}
			}
		}
	}
}

template<core::Device::DeviceType TDeviceType>
NNRT_CPU_OR_CUDA_DEVICE
inline bool FindAnchorsAndWeightsForPoint(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                          const int node_count, const Eigen::Vector3f& point,
                                          const NDArrayIndexer& node_indexer, const float node_coverage_squared) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	// region ===================== FIND ANCHOR POINTS ================================
	graph::FindKNNAnchorsBruteForce<TDeviceType>(anchor_indices, squared_distances, anchor_count,
	                                             node_count, point, node_indexer);
	// endregion
	// region ===================== COMPUTE ANCHOR WEIGHTS ================================
	float weight_sum = 0.0;
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float squared_distance = squared_distances[i_anchor];
		// note: equivalent to distance > 2 * node_coverage, avoids sqrtf
		if (squared_distance > 4 * node_coverage_squared) {
			anchor_indices[i_anchor] = -1;
			continue;
		}
		float weight = expf(-squared_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}
	if (valid_anchor_count < MINIMUM_VALID_ANCHOR_COUNT) {
		// TODO: verify
		//  a minimum of 1 node for fusion recommended by Neural Non-Rigid Tracking authors (?)
		return false;
	}
	if (weight_sum > 0.0f) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] /= weight_sum;
		}
	} else if (anchor_count > 0) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] = 1.0f / static_cast<float>(anchor_count);
		}
	}
	// endregion
	return true;
}

} // namespace graph
} // namespace kernel
} // namespace geometry
} // namespace nnrt