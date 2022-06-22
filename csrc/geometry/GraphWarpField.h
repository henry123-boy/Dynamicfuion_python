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

#include <open3d/t/geometry/TriangleMesh.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/core/TensorList.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

#include <pybind11/pybind11.h>
#include <open3d/core/Tensor.h>

#include "core/KdTree.h"
#include "geometry/kernel/WarpUtilities.h"

namespace py = pybind11;

namespace nnrt::geometry {

open3d::t::geometry::PointCloud
WarpPointCloud(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
               const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
               int anchor_count, float node_coverage,
               int minimum_valid_anchor_count = 0);

open3d::t::geometry::PointCloud
WarpPointCloud(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
               const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
               const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
               int minimum_valid_anchor_count = 0);

open3d::t::geometry::TriangleMesh WarpTriangleMesh(const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
                                                   const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                                                   int anchor_count, float node_coverage, bool threshold_nodes_by_distance = true,
                                                   int minimum_valid_anchor_count = 0);

void ComputeAnchorsAndWeightsEuclidean(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                       const open3d::core::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
                                       float node_coverage);

py::tuple ComputeAnchorsAndWeightsEuclidean(const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, int anchor_count,
                                            int minimum_valid_anchor_count, float node_coverage);

void ComputeAnchorsAndWeightsShortestPath(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                          const open3d::core::Tensor& nodes, const open3d::core::Tensor& edges, int anchor_count,
                                          float node_coverage);

py::tuple ComputeAnchorsAndWeightsShortestPath(const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
                                               const open3d::core::Tensor& edges, int anchor_count, float node_coverage);

class GraphWarpField {

public:
	GraphWarpField(open3d::core::Tensor nodes, open3d::core::Tensor edges, open3d::core::Tensor edge_weights, open3d::core::Tensor clusters,
	               float node_coverage = 0.05, bool threshold_nodes_by_distance_by_default = false, int anchor_count = 4, int minimum_valid_anchor_count = 0);
#ifdef __CUDACC__
	NNRT_HOST_DEVICE_WHEN_CUDACC
	GraphWarpField(const GraphWarpField& original) : nodes(original.nodes), edges(original.edges), edge_weights(original.edge_weights),
	clusters(original.clusters),

	translations(original.translations),
	rotations(original.rotations),


	node_coverage(original.node_coverage),
	anchor_count(original.anchor_count),
	threshold_nodes_by_distance_by_default(original.threshold_nodes_by_distance_by_default),
	minimum_valid_anchor_count(original.minimum_valid_anchor_count),

	index(original.index), node_coverage_squared(original.node_coverage_squared),
	kd_tree_nodes(original.kd_tree_nodes), kd_tree_node_count(original.kd_tree_node_count),
	node_indexer(original.node_indexer), rotation_indexer(original.rotation_indexer), translation_indexer(original.translation_indexer)

	{};
#else
	GraphWarpField(const GraphWarpField& original);
#endif
	virtual ~GraphWarpField() = default;

	open3d::core::Tensor GetWarpedNodes() const;
	open3d::core::Tensor GetNodeExtent() const;
	open3d::t::geometry::TriangleMesh WarpMesh(const open3d::t::geometry::TriangleMesh& input_mesh, bool disable_neighbor_thresholding = true) const;
	void ResetRotations();
	GraphWarpField ApplyTransformations() const;

	const core::KdTree& GetIndex() const;

	template<open3d::core::Device::DeviceType TDeviceType, bool UseNodeDistanceThreshold>
	NNRT_DEVICE_WHEN_CUDACC bool ComputeAnchorsForPoint(int32_t* anchor_indices, float* anchor_weights,
	                                               const Eigen::Vector3f& point) const {
		if (UseNodeDistanceThreshold) {
			return kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree_Threshold<TDeviceType>(
					anchor_indices, anchor_weights, anchor_count, minimum_valid_anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
					point, node_coverage_squared);
		} else {
			kernel::warp::FindAnchorsAndWeightsForPointEuclidean_KDTree<TDeviceType>(
					anchor_indices, anchor_weights, anchor_count, kd_tree_nodes, kd_tree_node_count, node_indexer,
					point, node_coverage_squared);
			return true;
		}
	}

	template<open3d::core::Device::DeviceType TDeviceType>
	NNRT_DEVICE_WHEN_CUDACC Eigen::Vector3f WarpPoint(const Eigen::Vector3f& point, const int32_t* anchor_indices, const float* anchor_weights) const{
		Eigen::Vector3f warped_point(0.f, 0.f, 0.f);
		kernel::warp::BlendWarp(warped_point, anchor_indices, anchor_weights, anchor_count, node_indexer,
		                        rotation_indexer, translation_indexer, point);
		return warped_point;
	}

	//TODO: gradually hide these fields and expose only on a need-to-know basis
	const open3d::core::Tensor nodes;
	const open3d::core::Tensor edges;
	const open3d::core::Tensor edge_weights;
	const open3d::core::Tensor clusters;

	open3d::core::Tensor translations;
	open3d::core::Tensor rotations;

	const float node_coverage;
	const int anchor_count;
	const bool threshold_nodes_by_distance_by_default;
	const int minimum_valid_anchor_count;

private:
	core::KdTree index;
	const float node_coverage_squared;
	const core::kernel::kdtree::KdTreeNode* kd_tree_nodes;
	const int32_t kd_tree_node_count;
	const open3d::t::geometry::kernel::NDArrayIndexer node_indexer;
	const open3d::t::geometry::kernel::NDArrayIndexer rotation_indexer;
	const open3d::t::geometry::kernel::NDArrayIndexer translation_indexer;

};


} // namespace nnrt::geometry