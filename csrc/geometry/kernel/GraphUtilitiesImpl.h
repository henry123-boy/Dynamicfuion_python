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

#include "geometry/kernel/Defines.h"
#include "core/DeviceHeap.h"


namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;

namespace nnrt {
namespace geometry {
namespace kernel {
namespace graph {

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void SetFloatsToValue(float* array, const int size, const float value){
	for (int i_anchor = 0; i_anchor < size; i_anchor++) {
		array[i_anchor] = value;
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void ProcessNodeEuclideanKNNBruteForce(int& max_at_index, float& max_squared_distance, int32_t* anchor_indices,
											  float* squared_distances, const int anchor_count,
											  const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer, const int i_node){
	auto node_pointer = node_indexer.GetDataPtrFromCoord<float>(i_node);
	Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
	float squared_distance = (node - point).squaredNorm();
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

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKNNAnchorsBruteForce(int32_t* anchor_indices, float* squared_distances, const int anchor_count,
                                              const int node_count, const Eigen::Vector3f& point,
                                              const NDArrayIndexer& node_indexer) {
	SetFloatsToValue<TDeviceType>(squared_distances, anchor_count, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	for (int i_node = 0; i_node < node_count; i_node++) {
		ProcessNodeEuclideanKNNBruteForce<TDeviceType>(max_at_index, max_squared_distance, anchor_indices, squared_distances, anchor_count,
													   point, node_indexer, i_node);
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindEuclideanKNNAnchorsBruteForce_ExcludingSet(int32_t* anchor_indices, float* squared_distances, const int anchor_count,
                                                           const int node_count, const Eigen::Vector3f& point,
                                                           const NDArrayIndexer& node_indexer, const int32_t* excluded_set, const int excluded_set_size) {
	SetFloatsToValue<TDeviceType>(squared_distances, anchor_count, INFINITY);

	int max_at_index = 0;
	float max_squared_distance = INFINITY;

	for (int i_node = 0; i_node < node_count; i_node++) {
		bool excluded = false;
		for(int i_excluded_index = 0; i_excluded_index < excluded_set_size; i_excluded_index++){
			if(i_node == excluded_set[i_excluded_index]){
				excluded = true;
			}
		}
		if(excluded) continue;

		ProcessNodeEuclideanKNNBruteForce<TDeviceType>(max_at_index, max_squared_distance, anchor_indices, squared_distances, anchor_count,
		                                               point, node_indexer, i_node);
	}
}

/**
 * \brief searches for anchor nodes within 2 * node_coverage of the specified point and computes their weights of influence on this point
 * \tparam TDeviceType
 * \param anchor_indices
 * \param anchor_weights
 * \param anchor_count
 * \param minimum_valid_anchor_count
 * \param node_count
 * \param point
 * \param node_indexer
 * \param node_coverage_squared
 * \return true if there are enough valid anchors within 2 * node_coverage, false otherwise
 */
template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPointEuclidean_Threshold(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                                 const int minimum_valid_anchor_count,
                                                 const int node_count, const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
                                                 const float node_coverage_squared) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	// region ===================== FIND ANCHOR POINTS ================================
	graph::FindEuclideanKNNAnchorsBruteForce<TDeviceType>(anchor_indices, squared_distances, anchor_count,
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
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	if (weight_sum > 0.0f) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] /= weight_sum;
		}
	} else if (valid_anchor_count > 0) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] = 1.0f / static_cast<float>(anchor_count);
		}
	}
	// endregion
	return true;
}


template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPointEuclidean(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                       const int node_count, const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer,
                                       const float node_coverage_squared) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	// region ===================== FIND ANCHOR POINTS ================================
	graph::FindEuclideanKNNAnchorsBruteForce<TDeviceType>(anchor_indices, squared_distances, anchor_count,
	                                                      node_count, point, node_indexer);
	// endregion
	// region ===================== COMPUTE ANCHOR WEIGHTS ================================

	float weight_sum = 0.0;
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float squared_distance = squared_distances[i_anchor];
		float weight = expf(-squared_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}

	if (weight_sum > 0.0f) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] /= weight_sum;
		}
	} else {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] = 1.0f / static_cast<float>(anchor_count);
		}
	}
	// endregion
}

/**
 * \brief find K nearest neighbors by shortest path distance within the provided graph.
 * Edges in the graph are expected to be in a N x GRAPH_DEGREE array, where each row contains indices of target nodes that share an edge with the
 * node at the row index.
 * \tparam TDeviceType
 * \param anchor_indices
 * \param distances
 * \param anchor_count
 * \param node_count
 * \param point
 * \param node_indexer
 * \param edge_indexer
 */
template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void FindShortestPathKNNAnchorsBruteForce(int32_t* anchor_indices, float* distances, const int anchor_count,
                                                 const int node_count, const Eigen::Vector3f& point,
                                                 const NDArrayIndexer& node_indexer,
                                                 const NDArrayIndexer& edge_indexer) {
	int discovered_anchor_count = 0;
	typedef core::KeyValuePair<float, int32_t> DistanceIndexPair;
	typedef decltype(core::MinHeapKeyCompare<float, int32_t>) Compare;

	const int queue_capacity = 40;
	DistanceIndexPair queue_data[queue_capacity];
	core::DeviceHeap<TDeviceType, DistanceIndexPair, Compare> priority_queue(queue_data, queue_capacity, core::MinHeapKeyCompare<float, int32_t>);
	while (discovered_anchor_count < anchor_count) {
		int closest_node_index = -1;
		float closest_node_distance;
		FindEuclideanKNNAnchorsBruteForce_ExcludingSet<TDeviceType>(&closest_node_index, &closest_node_distance, 1, node_count, point, node_indexer, anchor_indices, anchor_count);
		if(closest_node_index == -1){
			break; // no node to initialize queue with, we've got no more valid anchors to consider
		}
		priority_queue.insert(DistanceIndexPair{closest_node_distance, closest_node_index});

		while(!priority_queue.empty()){
			auto source_pair = priority_queue.pop();
			bool node_already_processed = false;
			for(int i_anchor = 0; i_anchor < discovered_anchor_count && !node_already_processed; i_anchor++){
				if(anchor_indices[i_anchor] == source_pair.key){
					node_already_processed = true;
				}
			}
			if(node_already_processed){
				continue;
			}
			distances[discovered_anchor_count] = source_pair.key;
			anchor_indices[discovered_anchor_count] = source_pair.value;
			discovered_anchor_count++;
			if(discovered_anchor_count >= anchor_count) break;

			auto source_pointer = node_indexer.GetDataPtrFromCoord<float>(source_pair.value);
			Eigen::Map<const Eigen::Vector3f> source_node(source_pointer);

			for(int i_edge = 0; i_edge < GRAPH_DEGREE; i_edge++){
				auto target_index_pointer = edge_indexer.template GetDataPtrFromCoord<int>(source_pair.value, i_edge);
				int target_node_index = *target_index_pointer;
				if(target_node_index > -1){
					auto target_pointer = node_indexer.GetDataPtrFromCoord<float>(target_node_index);
					Eigen::Map<const Eigen::Vector3f> target_node(target_pointer);
					float distance_source_to_target = (target_node - source_node).norm();
					priority_queue.insert(DistanceIndexPair{source_pair.value + distance_source_to_target, target_node_index});
				}
			}
		}
	}
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPointShortestPath(int32_t* anchor_indices, float* anchor_weights, const int anchor_count,
                                          const int node_count, const Eigen::Vector3f& point,
                                          const NDArrayIndexer& node_indexer,
                                          const NDArrayIndexer& edge_indexer,
                                          const float node_coverage_squared){
	auto distances = anchor_weights; // repurpose the anchor weights array to hold shortest path distances
	graph::FindShortestPathKNNAnchorsBruteForce<TDeviceType>(anchor_indices, distances, anchor_count, node_count, point, node_indexer, edge_indexer);
	// region ===================== COMPUTE ANCHOR WEIGHTS ================================

	float weight_sum = 0.0;
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float distance = distances[i_anchor];
		float weight = expf(-(distance*distance) / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}

	if (weight_sum > 0.0f) {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] /= weight_sum;
		}
	} else {
		for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
			anchor_weights[i_anchor] = 1.0f / static_cast<float>(anchor_count);
		}
	}
	// endregion
}


} // namespace graph
} // namespace kernel
} // namespace geometry
} // namespace nnrt