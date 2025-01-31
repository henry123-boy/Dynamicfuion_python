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
#include "geometry/WarpField.h"
#include "RegularizationLayer.h"


namespace nnrt::geometry {




class HierarchicalGraphWarpField : public WarpField {

public:
	HierarchicalGraphWarpField(
			open3d::core::Tensor nodes,
			float node_coverage = 0.05, // m
			bool threshold_nodes_by_distance_by_default = false,
			int anchor_count = 4,
			int minimum_valid_anchor_count = 0,
			WarpNodeCoverageComputationMethod warp_node_coverage_computation_method = WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE,
			int layer_count = 4,
			int max_vertex_degree = 4,
			std::function<float(int, float)> compute_layer_decimation_radius =
					[](int i_layer, float node_coverage){ return static_cast<float>(i_layer + 1) * node_coverage;}
	);


	void RebuildRegularizationLayers(int count, int max_vertex_degree);

	const RegularizationLayer& GetRegularizationLevel(int i_layer) const;
	int GetRegularizationLevelCount() const;

	const o3c::Tensor& GetNodePositions(bool use_virtual_ordering = false);
	const o3c::Tensor& GetNodeRotations(bool use_virtual_ordering = false);
	const o3c::Tensor& GetNodeTranslations(bool use_virtual_ordering = false);
	const o3c::Tensor& GetNodeCoverageWeights(bool use_virtual_ordering = false);

	const o3c::Tensor& GetEdges() const;
	const o3c::Tensor& GetVirtualNodeIndices() const;
	const o3c::Tensor& GetEdgeLayerIndices() const;
	const o3c::Tensor& GetLayerDecimationRadii() const;
	std::tuple<open3d::core::Tensor, open3d::core::Tensor> PrecomputeAnchorsAndWeights(
			const open3d::t::geometry::TriangleMesh& input_mesh,
			bool use_virtual_ordering = false
	);
	void TranslateNodes(const o3c::Tensor& node_translation_deltas, bool use_virtual_ordering = false);
	void RotateNodes(const o3c::Tensor& node_rotation_deltas, bool use_virtual_ordering = false);

private:
	class ReindexedTensorWrapper{
	public:
		explicit ReindexedTensorWrapper(o3c::Tensor& source_tensor);
		ReindexedTensorWrapper(const o3c::Tensor* index,  o3c::Tensor& source_tensor);
		const o3c::Tensor& Get(const o3c::Tensor* index);
		void Set(const o3c::Tensor& indexed_tensor);
	private:
		void Reindex();
		const o3c::Tensor* linear_index;
		o3c::Tensor& source_tensor;
		o3c::Tensor reindexed_tensor;
	};
	std::vector<RegularizationLayer> regularization_layers;
	std::function<float(int, float)> compute_layer_decimation_radius;
	o3c::Tensor edges;
	o3c::Tensor node_indices;
	o3c::Tensor layer_decimation_radii;
	open3d::core::Tensor edge_layer_indices;
	ReindexedTensorWrapper indexed_nodes;
	ReindexedTensorWrapper indexed_rotations;
	ReindexedTensorWrapper indexed_translations;
	ReindexedTensorWrapper indexed_node_coverage_weights;


};


} // namespace nnrt::geometry