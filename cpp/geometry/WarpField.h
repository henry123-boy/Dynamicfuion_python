//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/16/23.
//  Copyright (c) 2023 Gregory Kramida
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
// third-party includes
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>
// local includes
#include "core/KdTree.h"
#include "geometry/functional/kernel/WarpUtilities.h"
#include "WarpNodeCoverageComputationMethod.h"

namespace nnrt::geometry {
// TODO: there should be a separation of concerns between [node storage/retrieval + edge topology] and [warp anchor computation + warping].
//  This should address various TODOs throughout this header.
//  I'll need to rethink class hierarchy and overhaul implementation when I have time.
class WarpField {

public:
	WarpField(
			open3d::core::Tensor nodes,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0,
			WarpNodeCoverageComputationMethod warp_node_coverage_computation_method = WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE,
			int warp_node_coverage_neighbor_count = 4
	);
	WarpField(const WarpField& original) = default;
	WarpField(WarpField&& other) = default;

	virtual ~WarpField() = default;


	open3d::core::Tensor GetWarpedNodes() const;
	open3d::core::Tensor GetNodeExtent() const;
	open3d::t::geometry::TriangleMesh WarpMesh(
			const open3d::t::geometry::TriangleMesh& input_mesh, bool disable_neighbor_thresholding = true,
			const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))
	) const;
	open3d::t::geometry::TriangleMesh WarpMesh(
			const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& anchors,
			const open3d::core::Tensor& weights, bool disable_neighbor_thresholding = true,
			const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))) const;
	std::tuple<open3d::core::Tensor, open3d::core::Tensor> PrecomputeAnchorsAndWeights(
			const open3d::t::geometry::TriangleMesh& input_mesh
	) const;

	void ResetRotations();
	WarpField ApplyTransformations() const;

	const core::KdTree& GetIndex() const;

	OPEN3D_HOST_DEVICE Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> GetRotationForNode(int i_node) const {
		return Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(rotations_data + i_node * 9);
	}

	OPEN3D_HOST_DEVICE Eigen::Map<const Eigen::Vector3f> GetTranslationForNode(int i_node) const {
		return Eigen::Map<const Eigen::Vector3f>(translations_data + i_node * 3);
	}

	// TODO: take these two (ComputeAnchorsForPoint and WarpPoint) outside of class and declare as friend functions here, to take in a WarpField
	//  argument and perform the ops
	template<open3d::core::Device::DeviceType TDeviceType, bool UseNodeDistanceThreshold, bool UseFixedNodeCoverageWeight = false>
	NNRT_DEVICE_WHEN_CUDACC bool ComputeAnchorsForPoint(
			int32_t* anchor_indices, float* anchor_weights,
			const Eigen::Vector3f& point
	) const {
		if (UseFixedNodeCoverageWeight) {
			if (UseNodeDistanceThreshold) {
				return geometry::functional::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold_FixedNodeCoverageWeight<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
						point, node_coverage_squared);
			} else {
				geometry::functional::kernel::warp::FindAnchorsAndWeightsForPoint_Euclidean_KDTree_FixedNodeCoverageWeight<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
						point, node_coverage_squared);
				return true;
			}
		} else {
			if (UseNodeDistanceThreshold) {
				return geometry::functional::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold_VariableNodeCoverageWeight<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
						point, node_coverage_weight_data);
			} else {
				geometry::functional::kernel::warp::FindAnchorsAndWeightsForPoint_Euclidean_KDTree_VariableNodeCoverageWeight<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
						point, node_coverage_weight_data);
				return true;
			}
		}
	}


	template<open3d::core::Device::DeviceType TDeviceType>
	NNRT_DEVICE_WHEN_CUDACC Eigen::Vector3f
	WarpPoint(const Eigen::Vector3f& point, const int32_t* anchor_indices, const float* anchor_weights) const {
		Eigen::Vector3f warped_point(0.f, 0.f, 0.f);
		functional::kernel::warp::BlendWarp(
				warped_point, anchor_indices, anchor_weights, anchor_count, point, node_indexer,
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
					return GetRotationForNode(i_node);
				},
				NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int i_node) {
					return GetTranslationForNode(i_node);
				}
		);
		return warped_point;
	}

	WarpField Clone();


	//TODO: fix this mess, i.e. remove redundant methods, expose the direct const getter (and, if you still need, a setter) in python binding code
	// instead. Reference: https://github.com/pybind/pybind11/issues/141
	open3d::core::Tensor GetNodeRotations();
	const open3d::core::Tensor& GetNodeRotations() const;
	open3d::core::Tensor GetNodeTranslations();
	const open3d::core::Tensor& GetNodeTranslations() const;
	const open3d::core::Tensor& GetNodePositions() const;
	const open3d::core::Tensor& GetNodeCoverageWeights() const;
	open3d::core::Device GetDevice() const;

	void SetNodeRotations(const o3c::Tensor& node_rotations);
	void SetNodeTranslations(const o3c::Tensor& node_translations);

	void TranslateNodes(const o3c::Tensor& node_translation_deltas);
	void RotateNodes(const o3c::Tensor& node_rotation_deltas);

	open3d::core::Tensor node_positions;

	const float node_coverage;
	const int anchor_count;
	const bool threshold_nodes_by_distance_by_default;
	const int minimum_valid_anchor_count;
	const WarpNodeCoverageComputationMethod warp_node_coverage_computation_method;
	const int warp_node_coverage_neighbor_count;
protected:
	void RecomputeNodeCoverageWeights();
	WarpField(const WarpField& original, const core::KdTree& index);

	core::KdTree index;
	const float node_coverage_squared;

	const core::kernel::kdtree::KdTreeNode* kd_tree_nodes;
	const int32_t kd_tree_node_count;


	const open3d::t::geometry::kernel::NDArrayIndexer node_indexer;

	open3d::core::Tensor node_rotations;
	open3d::core::Tensor node_translations;
	open3d::core::Tensor node_coverage_weights;

	float const* rotations_data;
	float const* translations_data;
	float const* node_coverage_weight_data;

};

} // namespace nnrt::geometry