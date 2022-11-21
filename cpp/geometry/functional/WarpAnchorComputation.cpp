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

#include "geometry/kernel/Graph.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::geometry::functional {

void ComputeAnchorsAndWeightsEuclidean(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
                                       int anchor_count, int minimum_valid_anchor_count, float node_coverage) {
	auto device = points.GetDevice();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDevice(nodes, device);
	if (minimum_valid_anchor_count > anchor_count) {
		utility::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
		                  minimum_valid_anchor_count, anchor_count);
	}
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	auto points_shape = points.GetShape();
	if (points_shape.size() < 2 || points_shape.size() > 3) {
		utility::LogError("`points` needs to have 2 or 3 dimensions. Got: {} dimensions.", points_shape.size());
	}
	o3c::Tensor points_array;
	enum PointMode {
		POINT_ARRAY, POINT_IMAGE
	};
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
	kernel::graph::ComputeAnchorsAndWeightsEuclidean(anchors, weights, points_array, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
	if (point_mode == POINT_IMAGE) {
		anchors = anchors.Reshape({points_shape[0], points_shape[1], anchor_count});
		weights = weights.Reshape({points_shape[0], points_shape[1], anchor_count});
	}
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeightsEuclidean(const o3c::Tensor& points, const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
                                  float node_coverage) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
	return std::make_tuple(anchors, weights);
}

void ComputeAnchorsAndWeightsShortestPath(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
                                          const o3c::Tensor& edges, int anchor_count, float node_coverage) {
	auto device = points.GetDevice();
	o3c::AssertTensorDevice(nodes, device);
	o3c::AssertTensorDevice(edges, device);
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(edges, o3c::Dtype::Int32);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	auto points_shape = points.GetShape();
	if (points_shape.size() < 2 || points_shape.size() > 3) {
		utility::LogError("`points` needs to have 2 or 3 dimensions. Got: {} dimensions.", points_shape.size());
	}
	o3c::Tensor points_array;
	enum PointMode {
		POINT_ARRAY, POINT_IMAGE
	};
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
	kernel::graph::ComputeAnchorsAndWeightsShortestPath(anchors, weights, points_array, nodes, edges, anchor_count, node_coverage);
	if (point_mode == POINT_IMAGE) {
		anchors = anchors.Reshape({points_shape[0], points_shape[1], anchor_count});
		weights = weights.Reshape({points_shape[0], points_shape[1], anchor_count});
	}
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
ComputeAnchorsAndWeightsShortestPath(const o3c::Tensor& points, const o3c::Tensor& nodes, const o3c::Tensor& edges, int anchor_count,
                                     float node_coverage) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeightsShortestPath(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
	return std::make_tuple(anchors, weights);
}

} // namespace nnrt::geometry::functional