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
// third-party includes
// local includes
#include "geometry/WarpField.h"
#include "geometry/functional/Warping.h"
#include "core/linalg/Matmul3D.h"
#include "geometry/functional/WarpAnchorComputation.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry {

WarpField::WarpField(
		o3c::Tensor nodes,
		float node_coverage,
		bool threshold_nodes_by_distance_by_default,
		int anchor_count,
		int minimum_valid_anchor_count,
		WarpNodeCoverageComputationMethod warp_node_coverage_computation_method,
		int warp_node_coverage_neighbor_count
) :
		node_positions(std::move(nodes)), node_coverage(node_coverage), anchor_count(anchor_count),
		threshold_nodes_by_distance_by_default(threshold_nodes_by_distance_by_default),
		minimum_valid_anchor_count(minimum_valid_anchor_count),
		warp_node_coverage_computation_method(warp_node_coverage_computation_method),
		warp_node_coverage_neighbor_count(warp_node_coverage_neighbor_count),

		index(this->node_positions), node_coverage_squared(node_coverage * node_coverage),
		kd_tree_nodes(this->index.GetNodes()), kd_tree_node_count(this->index.GetNodeCount()),
		node_indexer(this->node_positions, 1),

		node_rotations({this->node_positions.GetLength(), 3, 3}, o3c::Dtype::Float32, this->node_positions.GetDevice()),
		node_translations(o3c::Tensor::Zeros({this->node_positions.GetLength(), 3}, o3c::Dtype::Float32, this->node_positions.GetDevice())),
		rotations_data(this->node_rotations.GetDataPtr<float>()),
		translations_data(this->node_translations.GetDataPtr<float>()),
		node_coverage_weight_data(nullptr) {

	int64_t node_count = this->node_positions.GetLength();
	o3c::AssertTensorShape(this->node_positions, { node_count, 3 });
	o3c::AssertTensorDtype(this->node_positions, o3c::Float32);
	if(node_count < anchor_count){
		utility::LogError("Anchor count for warp field, {}, exceeds node count, which is {}. "
						  "Anchors are nodes closest to a specific point using some distance metric, hence there cannot be more anchors than nodes.",
						  anchor_count, node_count);
	}

	this->ResetRotations();
	if (this->warp_node_coverage_computation_method == WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE) {
		RecomputeNodeCoverageWeights();
	}
}

WarpField::WarpField(const WarpField& original, const core::KdTree& index) :
		node_positions(index.GetPoints()),
		node_coverage(original.node_coverage), anchor_count(original.anchor_count),
		threshold_nodes_by_distance_by_default(original.threshold_nodes_by_distance_by_default),
		minimum_valid_anchor_count(original.minimum_valid_anchor_count),
		warp_node_coverage_computation_method(original.warp_node_coverage_computation_method),
		warp_node_coverage_neighbor_count(original.warp_node_coverage_neighbor_count),

		index(index),
		node_coverage_squared(original.node_coverage_squared),
		kd_tree_nodes(this->index.GetNodes()), kd_tree_node_count(this->index.GetNodeCount()),
		node_indexer(this->node_positions, 1),

		node_rotations(original.node_rotations.Clone()),
		node_translations(original.node_translations.Clone()),
		node_coverage_weights(original.node_coverage_weights.Clone()),
		rotations_data(this->node_rotations.GetDataPtr<float>()),
		translations_data(this->node_translations.GetDataPtr<float>()),
		node_coverage_weight_data(original.node_coverage_weight_data == nullptr ? nullptr : this->node_coverage_weights.GetDataPtr<float>()) {}

o3c::Tensor WarpField::GetWarpedNodes() const {
	return node_positions + this->node_translations;
}

o3c::Tensor WarpField::GetNodeExtent() const {
	o3c::Tensor minMax({2, 3}, node_positions.GetDtype(), node_positions.GetDevice());
	minMax.Slice(0, 0, 1) = node_positions.Min({0});
	minMax.Slice(0, 1, 2) = node_positions.Max({0});
	return minMax;
}

open3d::t::geometry::TriangleMesh
WarpField::WarpMesh(
		const open3d::t::geometry::TriangleMesh& input_mesh, bool disable_neighbor_thresholding,
		const open3d::core::Tensor& extrinsics/* = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) const {
	switch (this->warp_node_coverage_computation_method) {
		case WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE:
			if (disable_neighbor_thresholding) {
				return functional::WarpTriangleMesh(input_mesh, this->node_positions, this->node_rotations, this->node_translations,
				                                    this->anchor_count,
				                                    this->node_coverage,
				                                    false, 0, extrinsics);
			} else {
				return functional::WarpTriangleMesh(input_mesh, this->node_positions, this->node_rotations, this->node_translations,
				                                    this->anchor_count,
				                                    this->node_coverage,
				                                    this->threshold_nodes_by_distance_by_default, this->minimum_valid_anchor_count, extrinsics);
			}
			break;
		case WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE:
			utility::LogError("Not implemented", this->warp_node_coverage_computation_method);
			break;
		default: utility::LogError("Unsupported WarpNodeCoverageComputationMethod: {}", this->warp_node_coverage_computation_method);
			break;
	}


}

open3d::t::geometry::TriangleMesh
WarpField::WarpMesh(
		const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& anchors,
		const open3d::core::Tensor& weights, bool disable_neighbor_thresholding,
		const open3d::core::Tensor& extrinsics/* = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) const {
	if (disable_neighbor_thresholding) {
		return functional::WarpTriangleMeshUsingSuppliedAnchors(
				input_mesh, this->node_positions, this->node_rotations, this->node_translations, anchors, weights, false, 0, extrinsics);
	} else {
		return functional::WarpTriangleMeshUsingSuppliedAnchors(
				input_mesh, this->node_positions, this->node_rotations, this->node_translations, anchors,
				weights, this->threshold_nodes_by_distance_by_default, this->minimum_valid_anchor_count, extrinsics
		);
	}
}


const core::KdTree& WarpField::GetIndex() const {
	return this->index;
}

void WarpField::ResetRotations() {
	o3c::Tensor x;
	for (int i_node = 0; i_node < this->node_positions.GetLength(); i_node++) {
		node_rotations.Slice(0, i_node, i_node + 1) = o3c::Tensor::Eye(3, o3c::Dtype::Float32, this->node_positions.GetDevice());
	}
}

WarpField WarpField::ApplyTransformations() const {
	return {this->node_positions + this->node_translations, this->node_coverage,
	        this->threshold_nodes_by_distance_by_default, this->anchor_count, this->minimum_valid_anchor_count};
}

WarpField WarpField::Clone() {
	return {*this, this->index.Clone()};
}

open3d::core::Device WarpField::GetDevice() const {
	return this->node_positions.GetDevice();
}

void WarpField::SetNodeRotations(const o3c::Tensor& node_rotations) {
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorShape(node_rotations, { this->node_positions.GetLength(), 3, 3 });
	this->node_rotations = node_rotations;
	rotations_data = this->node_rotations.GetDataPtr<float>();
}

void WarpField::SetNodeTranslations(const o3c::Tensor& node_translations) {
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorShape(node_translations, { this->node_positions.GetLength(), 3 });
	this->node_translations = node_translations;
	translations_data = this->node_translations.GetDataPtr<float>();
}

void WarpField::TranslateNodes(const o3c::Tensor& node_translation_deltas) {
	this->node_translations += node_translation_deltas;
}

void WarpField::RotateNodes(const o3c::Tensor& node_rotation_deltas) {
	o3c::Tensor new_rotations;
	core::linalg::Matmul3D(new_rotations, this->node_rotations, node_rotation_deltas);
	this->node_rotations = new_rotations;
	rotations_data = this->node_rotations.GetDataPtr<float>();
}

open3d::core::Tensor WarpField::GetNodeRotations() {
	return this->node_rotations;
}

const open3d::core::Tensor& WarpField::GetNodeRotations() const {
	return node_rotations;
}

open3d::core::Tensor WarpField::GetNodeTranslations() {
	return this->node_translations;
}

const open3d::core::Tensor& WarpField::GetNodeTranslations() const {
	return this->node_translations;
}

const open3d::core::Tensor& WarpField::GetNodePositions() const {
	return node_positions;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor> WarpField::PrecomputeAnchorsAndWeights(
		const open3d::t::geometry::TriangleMesh& input_mesh
) const {
	o3c::Tensor anchors, weights;

	if (!input_mesh.HasVertexPositions()) {
		utility::LogError("Input mesh doesn't have vertex positions defined, which are required for computing warp field anchors & weights.");
	}
	const o3c::Tensor& vertex_positions = input_mesh.GetVertexPositions();
	switch (this->warp_node_coverage_computation_method) {
		case WarpNodeCoverageComputationMethod::FIXED_NODE_COVERAGE:
			functional::ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
					anchors, weights, vertex_positions, this->node_positions, this->anchor_count,
					this->minimum_valid_anchor_count, this->node_coverage
			);
			break;
		case WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE:
			functional::ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
					anchors, weights, vertex_positions, this->node_positions, this->node_coverage_weights, this->anchor_count,
					this->minimum_valid_anchor_count
			);
			break;
		default: utility::LogError("Unsupported WarpNodeCoverageComputationMethod: {}", this->warp_node_coverage_computation_method);
			break;
	}

	return std::make_tuple(anchors, weights);
}

const open3d::core::Tensor& WarpField::GetNodeCoverageWeights() const {
	return this->node_coverage_weights;
}

void WarpField::RecomputeNodeCoverageWeights() {
	if (this->warp_node_coverage_computation_method != WarpNodeCoverageComputationMethod::MINIMAL_K_NEIGHBOR_NODE_DISTANCE) {
		utility::LogError("Unsupported WarpNodeCoverageComputationMethod, {}, when attempting to call RecomputeNodeCoverageWeights",
		                  this->warp_node_coverage_computation_method);
	}
	if (this->node_positions.GetLength() == 1) {
		this->node_coverage_weights = o3c::Tensor(std::vector<float>{this->node_coverage}, {1}, o3c::Float32, this->GetDevice());
	} else {
		o3c::Tensor node_neighborhoods, distances;
		this->index.FindKNearestToPoints(node_neighborhoods, distances, node_positions, 2, true);
		this->node_coverage_weights = distances.Slice(1, 1, 2).Contiguous().Reshape({node_positions.GetLength()});
		this->node_coverage_weights = this->node_coverage_weights * this->node_coverage_weights;
	}
	this->node_coverage_weight_data = this->node_coverage_weights.GetDataPtr<float>();
}
} // namespace nnrt::geometry