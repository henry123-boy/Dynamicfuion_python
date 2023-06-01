//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/12/22.
//  Copyright (c) 2022 Gregory Kramida
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
#include "WarpAnchorComputation.h"

#include "geometry/functional/kernel/WarpAnchorComputation.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::geometry::functional {

enum PointMode {
	POINT_ARRAY, POINT_IMAGE
};

std::tuple<o3c::Tensor, o3c::SizeVector, PointMode> RunChecksFindModeAndPreparePoints(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		int anchor_count
) {
	auto device = points.GetDevice();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDevice(nodes, device);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	auto points_shape = points.GetShape();
	if (points_shape.size() < 2 || points_shape.size() > 3) {
		utility::LogError("`points` needs to have 2 or 3 dimensions. Got: {} dimensions.", points_shape.size());
	}
	o3c::Tensor points_array;

	PointMode point_mode;
	if (points_shape.size() == 2) {
		o3c::AssertTensorShape(points, { utility::nullopt, 3 });
		points_array = points;
		point_mode = POINT_ARRAY;
	} else {
		o3c::AssertTensorShape(points, { utility::nullopt, utility::nullopt, 3 });
		points_array = points.Reshape({-1, 3});
		point_mode = POINT_IMAGE;
	}
	return std::make_tuple(points_array, points_shape, point_mode);
}

void RestoreAnchorsAndWeightsToPointShape(
		o3c::Tensor& anchors,
		o3c::Tensor& weights,
		PointMode point_mode,
		const o3c::SizeVector& points_shape,
		int anchor_count
) {
	if (point_mode == POINT_IMAGE) {
		anchors = anchors.Reshape({points_shape[0], points_shape[1], anchor_count});
		weights = weights.Reshape({points_shape[0], points_shape[1], anchor_count});
	}
}

// region ======================================= EUCLIDEAN  =========================================================================================
void ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
) {
	o3c::Device device = points.GetDevice();
	int64_t node_count = nodes.GetShape(0);
	o3c::AssertTensorDevice(node_coverage_weights, device);
	o3c::AssertTensorDtype(node_coverage_weights, o3c::Float32);
	o3c::AssertTensorShape(node_coverage_weights, {node_count});

	o3c::Tensor points_array;
	o3c::SizeVector points_shape;
	PointMode point_mode;
	if (minimum_valid_anchor_count > anchor_count) {
		utility::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
		                  minimum_valid_anchor_count, anchor_count);
	}
	std::tie(points_array, points_shape, point_mode) = RunChecksFindModeAndPreparePoints(points, nodes, anchor_count);
	kernel::ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
			anchors, weights, points_array, nodes, node_coverage_weights, anchor_count, minimum_valid_anchor_count
	);
	RestoreAnchorsAndWeightsToPointShape(anchors, weights, point_mode, points_shape, anchor_count);
}


std::tuple<open3d::core::Tensor, open3d::core::Tensor> ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(anchors, weights, points, nodes, node_coverage_weights, anchor_count,
														  minimum_valid_anchor_count);
	return std::make_tuple(anchors, weights);
}

void ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		int anchor_count,
		int minimum_valid_anchor_count,
		float node_coverage_weight
) {
	o3c::Tensor points_array;
	o3c::SizeVector points_shape;
	PointMode point_mode;
	if (minimum_valid_anchor_count > anchor_count) {
		utility::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
		                  minimum_valid_anchor_count, anchor_count);
	}
	std::tie(points_array, points_shape, point_mode) = RunChecksFindModeAndPreparePoints(points, nodes, anchor_count);

	kernel::ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
			anchors, weights, points_array, nodes, anchor_count, minimum_valid_anchor_count,
			node_coverage_weight
	);
	RestoreAnchorsAndWeightsToPointShape(anchors, weights, point_mode, points_shape, anchor_count);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		int anchor_count,
		int minimum_valid_anchor_count,
		float node_coverage_weight
) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count,
	                                                   node_coverage_weight);
	return std::make_tuple(anchors, weights);
}
// endregion
// region ======================================= SHORTEST PATH ======================================================================================
void ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& weights,
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
) {
	auto device = points.GetDevice();
	o3c::AssertTensorDevice(edges, device);
	o3c::AssertTensorDtype(edges, o3c::Dtype::Int32);
	o3c::Tensor points_array;
	o3c::SizeVector points_shape;
	PointMode point_mode;
	std::tie(points_array, points_shape, point_mode) = RunChecksFindModeAndPreparePoints(points, nodes, anchor_count);
	kernel::ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(anchors, weights, points_array, nodes, node_coverage_weights, edges, anchor_count);
	RestoreAnchorsAndWeightsToPointShape(anchors, weights, point_mode, points_shape, anchor_count);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor> ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(anchors, weights, points, nodes, node_coverage_weights, edges, anchor_count);
	return std::make_tuple(anchors, weights);
}


void ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges, int anchor_count, float node_coverage_weight
) {
	auto device = points.GetDevice();
	o3c::AssertTensorDevice(edges, device);
	o3c::AssertTensorDtype(edges, o3c::Dtype::Int32);
	o3c::Tensor points_array;
	o3c::SizeVector points_shape;
	PointMode point_mode;
	std::tie(points_array, points_shape, point_mode) = RunChecksFindModeAndPreparePoints(points, nodes, anchor_count);
	kernel::ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(anchors, weights, points_array, nodes, edges, anchor_count, node_coverage_weight);
	RestoreAnchorsAndWeightsToPointShape(anchors, weights, point_mode, points_shape, anchor_count);
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, const open3d::core::Tensor& edges, int anchor_count,
		float node_coverage
) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
	return std::make_tuple(anchors, weights);
}

// endregion

} // namespace nnrt::geometry::functional