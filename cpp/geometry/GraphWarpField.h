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
// 3rd party
#include <pybind11/pybind11.h>
#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

// local
#include "core/KdTree.h"
#include "geometry/functional/Warping.h"
#include "geometry/functional/kernel/WarpUtilities.h"
#include "geometry/functional/AnchorComputationMethod.h"


namespace nnrt::geometry {


class WarpField {

public:
	WarpField(
			open3d::core::Tensor nodes,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0
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

	template<open3d::core::Device::DeviceType TDeviceType, bool UseNodeDistanceThreshold>
	NNRT_DEVICE_WHEN_CUDACC bool ComputeAnchorsForPoint(
			int32_t* anchor_indices, float* anchor_weights,
			const Eigen::Vector3f& point
	) const {
		if (UseNodeDistanceThreshold) {
			return geometry::functional::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold<TDeviceType>(
					anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
					point, node_coverage_squared);
		} else {
			geometry::functional::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree<TDeviceType>(
					anchor_indices, anchor_weights, anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
					point, node_coverage_squared);
			return true;
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


	open3d::core::Tensor GetNodeRotations();
	const open3d::core::Tensor& GetNodeRotations() const;
	open3d::core::Tensor GetNodeTranslations();
	const open3d::core::Tensor& GetNodeTranslations() const;
	const open3d::core::Tensor& GetNodePositions() const;
	open3d::core::Device GetDevice() const;

	void SetNodeRotations(const o3c::Tensor& node_rotations);
	void SetNodeTranslations(const o3c::Tensor& node_translations);

	void TranslateNodes(const o3c::Tensor& node_translation_deltas);
	void RotateNodes(const o3c::Tensor& node_rotation_deltas);

	const open3d::core::Tensor nodes;

	const float node_coverage;
	const int anchor_count;
	const bool threshold_nodes_by_distance_by_default;
	const int minimum_valid_anchor_count;
protected:
	WarpField(const WarpField& original, const core::KdTree& index);

	core::KdTree index;
	const float node_coverage_squared;
	const core::kernel::kdtree::KdTreeNode* kd_tree_nodes;
	const int32_t kd_tree_node_count;


	const open3d::t::geometry::kernel::NDArrayIndexer node_indexer;

	open3d::core::Tensor rotations;
	open3d::core::Tensor translations;

	float const* rotations_data;
	float const* translations_data;

};

class PlanarGraphWarpField : public WarpField {
public:
	PlanarGraphWarpField(
			open3d::core::Tensor nodes,
			open3d::core::Tensor edges,
			open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights,
			open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0
	);
	PlanarGraphWarpField(const PlanarGraphWarpField& original) = default;
	PlanarGraphWarpField(PlanarGraphWarpField&& other) = default;

	std::tuple<open3d::core::Tensor, open3d::core::Tensor> PrecomputeAnchorsAndWeights(
			const open3d::t::geometry::TriangleMesh& input_mesh,
			AnchorComputationMethod anchor_computation_method
	) const;

	const open3d::core::Tensor edges;
	open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> edge_weights;
	open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> clusters;

	PlanarGraphWarpField ApplyTransformations() const;

protected:
	PlanarGraphWarpField(const PlanarGraphWarpField& original, const core::KdTree& index);

};


class HierarchicalGraphWarpField : public WarpField {
public:
	struct RegularizationLayer {
	public:
		float node_coverage; // m
		open3d::core::Tensor nodes;
		open3d::core::Tensor edges_to_coarser_layer;
		open3d::core::Tensor rotations;
		open3d::core::Tensor translations;
		std::shared_ptr<core::KdTree> index;
	};
public:
	HierarchicalGraphWarpField(
			open3d::core::Tensor nodes,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0,
			int layer_count = 4,
			int max_vertex_degree = 4,
			std::function<float(int, float)> compute_layer_decimation_radius =
					[](int i_layer, float node_coverage){ return static_cast<float>(i_layer + 1) * node_coverage;}
	);


	void RebuildRegularizationLayers(int count, int max_vertex_degree);

	const HierarchicalGraphWarpField::RegularizationLayer& GetRegularizationLevel(int i_layer) const;
	int GetRegularizationLevelCount() const;

protected:
	std::vector<RegularizationLayer> regularization_layers;
	std::function<float(int, float)> compute_layer_decimation_radius;

};


} // namespace nnrt::geometry