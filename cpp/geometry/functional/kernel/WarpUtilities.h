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

#include "geometry/functional/kernel/Defines.h"
#include "core/kernel/KnnUtilities.h"
#include "core/kernel/KnnUtilities_PriorityQueue.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/heap/DeviceHeap.h"


namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
namespace kdtree = nnrt::core::kernel::kdtree;

namespace nnrt::geometry::functional::kernel::warp {

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
ComputeAnchorWeights_VariableNodeCoverageWeight(
		float* anchor_weights,
		float& weight_sum,
		const int* anchor_node_indices,
		const float* node_weights,
		const float* distances,
		const int anchor_count
) {
	weight_sum = 0.0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float square_distance = distances[i_anchor];
		if (!TDistancesAreSquared) {
			square_distance = square_distance * square_distance;
		}
		float node_coverage_squared = node_weights[anchor_node_indices[i_anchor]];
		float weight = expf(-square_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
	}
}

template<o3c::Device::DeviceType TDeviceType, bool TDistancesAreSquared>
NNRT_DEVICE_WHEN_CUDACC
inline void
ComputeAnchorWeights_FixedNodeCoverageWeight(
		float* anchor_weights,
		float& weight_sum,
		const float* distances,
		const int anchor_count,
		const float node_coverage_squared
) {
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
ComputeAnchorWeights_Threshold_VariableNodeCoverageWeight(
		int* anchor_node_indices,
		float* anchor_weights,
		float& weight_sum,
		const float* node_weights,
		int& valid_anchor_count,
		const float* distances,
		const int anchor_count
) {
	weight_sum = 0.0;
	valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float square_distance = distances[i_anchor];

		if (!TDistancesAreSquared) {
			square_distance = square_distance * square_distance;
		}
		float node_coverage_squared = node_weights[anchor_node_indices[i_anchor]];
		// note: equivalent to distance > 2 * node_coverage, avoids sqrtf
		if (square_distance > 4 * node_coverage_squared) {
			anchor_node_indices[i_anchor] = -1;
			continue;
		}
		float weight = expf(-square_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}
}

template<o3c::Device::DeviceType TDeviceType, bool TDistancesAreSquared>
NNRT_DEVICE_WHEN_CUDACC
inline void
ComputeAnchorWeights_Threshold_FixedNodeCoverageWeight(
		int* anchor_node_indices, float* anchor_weights, float& weight_sum, int& valid_anchor_count, const float* distances,
		const int anchor_count,
		const float node_coverage_squared
) {
	weight_sum = 0.0;
	valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		float square_distance = distances[i_anchor];
		if (!TDistancesAreSquared) {
			square_distance = square_distance * square_distance;
		}
		// note: equivalent to distance > 2 * node_coverage, avoids sqrtf
		if (square_distance > 4 * node_coverage_squared) {
			anchor_node_indices[i_anchor] = -1;
			continue;
		}
		float weight = expf(-square_distance / (2 * node_coverage_squared));
		weight_sum += weight;
		anchor_weights[i_anchor] = weight;
		valid_anchor_count++;
	}
}

//@formatter:off
template<o3c::Device::DeviceType TDeviceType, typename TPointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPoint_Euclidean_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices,
		float* anchor_weights,

		const int anchor_count,
		const int node_count,
		TPointVector& point,
		const NDArrayIndexer& node_indexer,
		const float* node_weights
) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKnn_BruteForce<TDeviceType>(anchor_node_indices, squared_distances, anchor_count, node_count, point, node_indexer);
	float weight_sum;
	ComputeAnchorWeights_VariableNodeCoverageWeight<TDeviceType, true>(
			anchor_weights, weight_sum, anchor_node_indices, node_weights, squared_distances, anchor_count
	);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}
template<o3c::Device::DeviceType TDeviceType, typename TPointVector>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPoint_Euclidean_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices,
		float* anchor_weights,

		const int anchor_count,
		const int node_count,
		TPointVector& point,
		const NDArrayIndexer& node_indexer,
		const float node_coverage_squared
) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKnn_BruteForce<TDeviceType>(anchor_node_indices, squared_distances, anchor_count, node_count, point, node_indexer
	);
	float weight_sum;
	ComputeAnchorWeights_FixedNodeCoverageWeight<TDeviceType, true>(
			anchor_weights, weight_sum, squared_distances, anchor_count,
			node_coverage_squared
	);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}
//@formatter:on



template<o3c::Device::DeviceType TDeviceType, typename TPointVector>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPoint_Euclidean_Threshold_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const int node_count, TPointVector& point,
		const NDArrayIndexer& node_indexer, const float* node_weights
) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKnn_BruteForce<TDeviceType>(anchor_node_indices, squared_distances, anchor_count, node_count, point, node_indexer);
	float weight_sum;
	int valid_anchor_count;

	ComputeAnchorWeights_Threshold_VariableNodeCoverageWeight<TDeviceType, true>(
			anchor_node_indices, anchor_weights, weight_sum, node_weights, valid_anchor_count, squared_distances, anchor_count
	);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, valid_anchor_count);
	return true;
}

template<o3c::Device::DeviceType TDeviceType, typename TPointVector>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPoint_Euclidean_Threshold_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const int node_count, TPointVector& point,
		const NDArrayIndexer& node_indexer, const float node_coverage_squared
) {
	auto squared_distances = anchor_weights; // repurpose the anchor weights array to hold squared distances
	core::kernel::knn::FindEuclideanKnn_BruteForce<TDeviceType>(anchor_node_indices, squared_distances, anchor_count, node_count, point, node_indexer);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold_FixedNodeCoverageWeight<TDeviceType, true>(anchor_node_indices, anchor_weights, weight_sum, valid_anchor_count,
	                                                                          squared_distances, anchor_count,
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
FindAnchorsAndWeightsForPoint_Euclidean_KDTree_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const kdtree::KdTreeNode* kd_tree_nodes,
		const int kd_tree_node_count, const NDArrayIndexer& node_indexer,
		const Eigen::Vector3f& point, const float* node_weights
) {
	auto anchor_distances = anchor_weights; // repurpose the anchor weights array to hold anchor distances
	// note that in this function, some nomenclature gets flipped around. Whereas we're still calling the motion graph control nodes as "nodes" here,
	// in the KdTree context, KD Tree nodes become more relevant, whereas the old "graph" nodes are called "reference points".
	core::kernel::knn::FindEuclideanKnn_KdTree<TDeviceType>(anchor_node_indices, anchor_distances, kd_tree_nodes, kd_tree_node_count, anchor_count, point,
	                                                        node_indexer);
	float weight_sum;
	ComputeAnchorWeights_VariableNodeCoverageWeight<TDeviceType, false>(anchor_weights, weight_sum, anchor_node_indices,
	                                                                    node_weights, anchor_distances, anchor_count);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPoint_Euclidean_KDTree_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const kdtree::KdTreeNode* kd_tree_nodes,
		const int kd_tree_node_count, const NDArrayIndexer& node_indexer,
		const Eigen::Vector3f& point, const float node_coverage_squared
) {
	auto anchor_distances = anchor_weights; // repurpose the anchor weights array to hold anchor distances
	// note that in this function, some nomenclature gets flipped around. Whereas we're still calling the motion graph control nodes as "nodes" here,
	// in the KdTree context, KD Tree nodes become more relevant, whereas the old "graph" nodes are called "reference points".
	core::kernel::knn::FindEuclideanKnn_KdTree<TDeviceType>(anchor_node_indices, anchor_distances, kd_tree_nodes, kd_tree_node_count, anchor_count, point,
	                                                        node_indexer);
	float weight_sum;
	ComputeAnchorWeights_FixedNodeCoverageWeight<TDeviceType, false>(anchor_weights, weight_sum, anchor_distances, anchor_count,
	                                                                 node_coverage_squared);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const kdtree::KdTreeNode* kd_tree_nodes,
		const int kd_tree_node_count, const NDArrayIndexer& node_indexer,
		const Eigen::Vector3f& point, const float* node_weights
) {
	auto anchor_distances = anchor_weights; // repurpose the anchor weights array to hold anchor distances
	// note that in this function, some nomenclature gets flipped around. Whereas we're still calling the motion graph control nodes as "nodes" here,
	// in the KdTree context, KD Tree nodes become more relevant, whereas the old "graph" nodes are called "reference points".
	core::kernel::knn::FindEuclideanKnn_KdTree<TDeviceType>(anchor_node_indices, anchor_distances, kd_tree_nodes, kd_tree_node_count, anchor_count, point,
	                                                        node_indexer);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold_VariableNodeCoverageWeight<TDeviceType, false>(
			anchor_node_indices, anchor_weights, weight_sum, node_weights, valid_anchor_count, anchor_distances, anchor_count
	);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, valid_anchor_count);
	return true;
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const kdtree::KdTreeNode* kd_tree_nodes,
		const int kd_tree_node_count, const NDArrayIndexer& node_indexer,
		const Eigen::Vector3f& point, const float node_coverage_squared
) {
	auto anchor_distances = anchor_weights; // repurpose the anchor weights array to hold anchor distances
	// note that in this function, some nomenclature gets flipped around. Whereas we're still calling the motion graph control nodes as "nodes" here,
	// in the KdTree context, KD Tree nodes become more relevant, whereas the old "graph" nodes are called "reference points".
	core::kernel::knn::FindEuclideanKnn_KdTree<TDeviceType>(anchor_node_indices, anchor_distances, kd_tree_nodes, kd_tree_node_count, anchor_count, point,
	                                                        node_indexer);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold_FixedNodeCoverageWeight<TDeviceType, false>(
			anchor_node_indices, anchor_weights, weight_sum, valid_anchor_count,
			anchor_distances, anchor_count, node_coverage_squared
	);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, valid_anchor_count);
	return true;
}


template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPoint_ShortestPath_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count, const int node_count,
		const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer,
		const float* node_weights, const int graph_degree
) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold the shortest path distances to anchors
	core::kernel::knn::FindShortestPathKnn_PriorityQueue<TDeviceType>(anchor_node_indices, distances, anchor_count, node_count, point, node_indexer,
	                                                                  edge_indexer, graph_degree);
	float weight_sum = 0.0;
	ComputeAnchorWeights_VariableNodeCoverageWeight<TDeviceType, false>(anchor_weights, weight_sum, anchor_node_indices, node_weights, distances,
	                                                                    anchor_count);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline void
FindAnchorsAndWeightsForPoint_ShortestPath_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count, const int node_count,
		const Eigen::Vector3f& point, const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer,
		const float node_coverage_squared, const int graph_degree
) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold the shortest path distances to anchors
	core::kernel::knn::FindShortestPathKnn_PriorityQueue<TDeviceType>(anchor_node_indices, distances, anchor_count, node_count, point, node_indexer,
	                                                                  edge_indexer, graph_degree);
	float weight_sum = 0.0;
	ComputeAnchorWeights_FixedNodeCoverageWeight<TDeviceType, false>(anchor_weights, weight_sum, distances, anchor_count, node_coverage_squared);
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
}


template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPoint_ShortestPath_Threshold_VariableNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const int node_count, const Eigen::Vector3f& point,
		const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer,
		const float* node_weights, const int graph_degree
) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold shortest path distances
	core::kernel::knn::FindShortestPathKnn_PriorityQueue<TDeviceType>(anchor_node_indices, distances, anchor_count, node_count, point, node_indexer,
	                                                                  edge_indexer, graph_degree);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold_VariableNodeCoverageWeight<TDeviceType, false>(
			anchor_node_indices, anchor_weights, weight_sum, node_weights, valid_anchor_count, distances, anchor_count
	);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
	return true;
}

template<o3c::Device::DeviceType TDeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline bool
FindAnchorsAndWeightsForPoint_ShortestPath_Threshold_FixedNodeCoverageWeight(
		int32_t* anchor_node_indices, float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const int node_count, const Eigen::Vector3f& point,
		const NDArrayIndexer& node_indexer, const NDArrayIndexer& edge_indexer,
		const float node_coverage_squared, const int graph_degree
) {
	auto distances = anchor_weights; // repurpose the anchor weights array to hold shortest path distances
	core::kernel::knn::FindShortestPathKnn_PriorityQueue<TDeviceType>(anchor_node_indices, distances, anchor_count, node_count, point, node_indexer,
	                                                                  edge_indexer, graph_degree);
	float weight_sum;
	int valid_anchor_count;
	ComputeAnchorWeights_Threshold_FixedNodeCoverageWeight<TDeviceType, false>(anchor_node_indices, anchor_weights, weight_sum, valid_anchor_count,
	                                                                           distances, anchor_count,
	                                                                           node_coverage_squared);
	if (valid_anchor_count < minimum_valid_anchor_count) {
		return false;
	}
	NormalizeAnchorWeights<TDeviceType>(anchor_weights, weight_sum, anchor_count, anchor_count);
	return true;
}

// point, generic retrieval
template<typename TPoint, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp(
		TPoint& warped_point, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const Eigen::Vector3f& source_point, const NDArrayIndexer& node_indexer, TGetRotationForNode&& get_rotation_for_node,
		TGetTranslationForNode&& get_translation_for_node
) {
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		int anchor_node_index = anchor_node_indices[i_anchor];
		if (anchor_node_index != -1) {
			float anchor_weight = anchor_weights[i_anchor];
			auto node_rotation = get_rotation_for_node(anchor_node_index);
			auto node_translation = get_translation_for_node(anchor_node_index);
			auto node_pointer = node_indexer.GetDataPtr<float>(anchor_node_index);
			Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
			warped_point += anchor_weight * (node + node_rotation * (source_point - node) + node_translation);
		}
	}
}

// point + normal, generic retrieval
template<typename TPoint, typename TNormal, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp(
		TPoint& warped_point, TNormal& warped_normal, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const Eigen::Vector3f& source_point, const Eigen::Vector3f& source_normal, const NDArrayIndexer& node_indexer,
		TGetRotationForNode&& get_rotation_for_node, TGetTranslationForNode&& get_translation_for_node
) {
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		int anchor_node_index = anchor_node_indices[i_anchor];
		if (anchor_node_index != -1) {
			float anchor_weight = anchor_weights[i_anchor];
			auto node_rotation = get_rotation_for_node(anchor_node_index);
			auto node_translation = get_translation_for_node(anchor_node_index);
			auto node_pointer = node_indexer.GetDataPtr<float>(anchor_node_index);
			Eigen::Vector3f node(node_pointer[0], node_pointer[1], node_pointer[2]);
			warped_point += anchor_weight * (node + node_rotation * (source_point - node) + node_translation);
			warped_normal += anchor_weight * (node_rotation * source_normal);
		}
	}
}

// point, indexers
template<typename TPoint>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp(
		TPoint& warped_point, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const Eigen::Vector3f& source_point, const NDArrayIndexer& node_indexer, const NDArrayIndexer& node_rotation_indexer,
		const NDArrayIndexer& node_translation_indexer
) {
	BlendWarp(warped_point, anchor_node_indices, anchor_weights, anchor_count, source_point, node_indexer,
	          NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
		          return Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(node_rotation_indexer.GetDataPtr<float>(i_node));
	          },
	          NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
		          return Eigen::Map<const Eigen::Vector3f>(node_translation_indexer.GetDataPtr<float>(i_node));
	          }
	);
}

// point + normal, indexers
template<typename TPoint, typename TNormal>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp(
		TPoint& warped_point, TNormal& warped_normal, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const Eigen::Vector3f& source_point, const Eigen::Vector3f& source_normal, const NDArrayIndexer& node_indexer,
		const NDArrayIndexer& node_rotation_indexer, const NDArrayIndexer& node_translation_indexer
) {
	BlendWarp(warped_point, warped_normal, anchor_node_indices, anchor_weights, anchor_count, source_point, source_normal, node_indexer,
	          NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
		          return Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(node_rotation_indexer.GetDataPtr<float>(i_node));
	          },
	          NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
		          return Eigen::Map<const Eigen::Vector3f>(node_translation_indexer.GetDataPtr<float>(i_node));
	          }
	);
}

// point, generic retrieval
template<typename TPoint, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp_ValidAnchorCountThreshold(
		TPoint& warped_point, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const Eigen::Vector3f& source_point,
		const NDArrayIndexer& node_indexer, TGetRotationForNode&& get_rotation_for_node,
		TGetTranslationForNode&& get_translation_for_node
) {
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		valid_anchor_count += static_cast<int>(anchor_node_indices[i_anchor] != -1);
	}
	if (valid_anchor_count >= minimum_valid_anchor_count) {
		BlendWarp(warped_point, anchor_node_indices, anchor_weights, anchor_count, source_point,
		          node_indexer, get_rotation_for_node, get_translation_for_node);
	}
}

// point + normal, generic retrieval
template<typename TPoint, typename TNormal, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp_ValidAnchorCountThreshold(
		TPoint& warped_point, TNormal& warped_normal, const int32_t* anchor_node_indices, const float* anchor_weights,
		const int anchor_count, const int minimum_valid_anchor_count, const Eigen::Vector3f& source_point,
		const Eigen::Vector3f& source_normal, const NDArrayIndexer& node_indexer,
		TGetRotationForNode&& get_rotation_for_node, TGetTranslationForNode&& get_translation_for_node
) {
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		valid_anchor_count += static_cast<int>(anchor_node_indices[i_anchor] != -1);
	}
	if (valid_anchor_count >= minimum_valid_anchor_count) {
		BlendWarp(warped_point, warped_normal, anchor_node_indices, anchor_weights, anchor_count, source_point, source_normal, node_indexer,
		          get_rotation_for_node, get_translation_for_node);
	}
}

// point, indexers
template<typename TPoint, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp_ValidAnchorCountThreshold(
		TPoint& warped_point, const int32_t* anchor_node_indices, const float* anchor_weights, const int anchor_count,
		const int minimum_valid_anchor_count, const Eigen::Vector3f& source_point,
		const NDArrayIndexer& node_indexer, const NDArrayIndexer& node_rotation_indexer,
		const NDArrayIndexer& node_translation_indexer
) {
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		valid_anchor_count += static_cast<int>(anchor_node_indices[i_anchor] != -1);
	}
	if (valid_anchor_count >= minimum_valid_anchor_count) {
		BlendWarp(warped_point, anchor_node_indices, anchor_weights, anchor_count,
		          node_rotation_indexer, node_indexer, node_translation_indexer, source_point);
	}
}

// point + normal, indexers
template<typename TPoint, typename TNormal, typename TGetRotationForNode, typename TGetTranslationForNode>
inline NNRT_DEVICE_WHEN_CUDACC
void BlendWarp_ValidAnchorCountThreshold(
		TPoint& warped_point, TNormal& warped_normal, const int32_t* anchor_node_indices, const float* anchor_weights,
		const int anchor_count, const int minimum_valid_anchor_count,
		const Eigen::Vector3f& source_point, const Eigen::Vector3f& source_normal,
		const NDArrayIndexer& node_indexer,
		const NDArrayIndexer& node_rotation_indexer, const NDArrayIndexer& node_translation_indexer
) {
	int valid_anchor_count = 0;
	for (int i_anchor = 0; i_anchor < anchor_count; i_anchor++) {
		valid_anchor_count += static_cast<int>(anchor_node_indices[i_anchor] != -1);
	}
	if (valid_anchor_count >= minimum_valid_anchor_count) {
		BlendWarp(warped_point, warped_normal, anchor_node_indices, anchor_weights, anchor_count, source_point, source_normal, node_indexer,
		          node_rotation_indexer, node_translation_indexer);
	}
}


} // namespace nnrt::geometry::kernel::warp