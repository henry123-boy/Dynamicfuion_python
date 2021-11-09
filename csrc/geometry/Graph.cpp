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
#include "geometry/Graph.h"
#include "geometry/kernel/Graph.h"
#include "geometry/kernel/Warp.h"

using namespace open3d;
using namespace open3d::t::geometry;

namespace nnrt::geometry {

void CheckNodeData(core::Device& device, const core::Tensor& nodes, const core::Tensor& node_rotations,
                   const core::Tensor& node_translations) {
	if (device != nodes.GetDevice() || device != node_rotations.GetDevice() || device != node_translations.GetDevice()) {
		utility::LogError("Device not consistent among arguments.");
	}
	auto nodes_shape = nodes.GetShape();
	auto rotations_shape = node_rotations.GetShape();
	auto translations_shape = node_translations.GetShape();
	if (nodes_shape.size() != 2 || rotations_shape.size() != 3 || translations_shape.size() != 2) {
		utility::LogError("Arguments nodes, rotations, and translations need to have 2, 3, and 2 dimensions,"
		                  " respectively. Got {}, {}, and {}.", nodes_shape.size(),
		                  rotations_shape.size(), translations_shape.size());
	}

	const int64_t node_count = nodes_shape[0];
	if (nodes_shape[1] != 3) {
		utility::LogError("Argument nodes needs to have size N x 3, has size N x {}.", nodes_shape[1]);
	}
	if (rotations_shape[0] != node_count || rotations_shape[1] != 3 || rotations_shape[2] != 3) {
		utility::LogError("Argument node_rotations needs to have shape ({}, 3, 3), where first dimension is the node count N"
		                  ", but has shape {}", node_count, rotations_shape);
	}
	if (translations_shape[0] != node_count || translations_shape[1] != 3) {
		utility::LogError("Argument node_translations needs to have shape ({}, 3), where first dimension is the node count N"
		                  ", but has shape {}", node_count, translations_shape);
	}
	if (device != nodes.GetDevice() ||
	    device != node_rotations.GetDevice() ||
	    device != node_translations.GetDevice()) {
		utility::LogError("Inputs' devices don't match.", node_count, translations_shape);
	}
	nodes.AssertDtype(core::Dtype::Float32);
	node_rotations.AssertDtype(core::Dtype::Float32);
	node_translations.AssertDtype(core::Dtype::Float32);
}

PointCloud
WarpPointCloudMat(const PointCloud& input_point_cloud, const core::Tensor& nodes, const core::Tensor& node_rotations,
                  const core::Tensor& node_translations, const int anchor_count, float node_coverage,
                  int minimum_valid_anchor_count) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeData(device, nodes, node_rotations, node_translations);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		utility::LogError("`anchor_count` is {}, but is required to satisfy 0 < anchor_count <= {}", anchor_count, MAX_ANCHOR_COUNT);
	}
	if (minimum_valid_anchor_count < 0 || minimum_valid_anchor_count > anchor_count) {
		utility::LogError("`minimum_valid_anchor_count` is {}, but is required to satisfy 0 < minimum_valid_anchor_count <= {} ",
		                  minimum_valid_anchor_count, anchor_count);
	}
	// endregion

	PointCloud warped_point_cloud(device);

	if (warped_point_cloud.HasPointColors()) {
		warped_point_cloud.SetPointColors(input_point_cloud.GetPointColors());
	}


	if (input_point_cloud.HasPoints()) {
		const auto& vertices = input_point_cloud.GetPoints();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		vertices.AssertDtype(core::Dtype::Float32);
		core::Tensor warped_points;
		kernel::warp::WarpPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
		                         minimum_valid_anchor_count);
		warped_point_cloud.SetPoints(warped_points);
	}

	return warped_point_cloud;
}


PointCloud
WarpPointCloudMat(const PointCloud& input_point_cloud, const core::Tensor& nodes, const core::Tensor& node_rotations,
                  const core::Tensor& node_translations, const core::Tensor& anchors, const core::Tensor& anchor_weights,
                  int minimum_valid_anchor_count) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeData(device, nodes, node_rotations, node_translations);
	auto anchors_shape = anchors.GetShape();
	auto anchor_weights_shape = anchor_weights.GetShape();
	if (anchors_shape.size() != 2 || anchor_weights_shape.size() != 2) {
		utility::LogError("Tensors `anchors` and `anchor_weights` need to both have two dimensions."
		                  "Got {} and {} dimensions, respectively.", anchors_shape.size(),
		                  anchor_weights_shape.size());
	}
	if (anchors_shape[0] != anchor_weights_shape[0] || anchors_shape[1] != anchor_weights_shape[1]) {
		utility::LogError("Tensors `anchors` and `anchor_weights` need to have matching dimensions."
		                  "Got {} and {}, respectively.", anchors_shape,
		                  anchor_weights_shape);
	}
	const int64_t anchor_count = anchors_shape[1];
	if (minimum_valid_anchor_count < 0 || minimum_valid_anchor_count > anchor_count) {
		utility::LogError("`minimum_valid_anchor_count` is {}, but is required to satisfy 0 < minimum_valid_anchor_count <= {}, "
		                  "where the upper bound is the second dimension of the input `anchors` tensor.",
		                  minimum_valid_anchor_count, anchor_count);
	}
	anchors.AssertDtype(core::Dtype::Int32);
	anchor_weights.AssertDtype(core::Dtype::Float32);

	PointCloud warped_point_cloud(device);

	if (warped_point_cloud.HasPointColors()) {
		warped_point_cloud.SetPointColors(input_point_cloud.GetPointColors());
	}


	if (input_point_cloud.HasPoints()) {
		const auto& vertices = input_point_cloud.GetPoints();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		vertices.AssertDtype(core::Dtype::Float32);
		core::Tensor warped_points;
		kernel::warp::WarpPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchors, anchor_weights,
		                         minimum_valid_anchor_count);
		warped_point_cloud.SetPoints(warped_points);
	}

	return warped_point_cloud;
}


open3d::t::geometry::TriangleMesh
WarpTriangleMeshMat(const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
                    const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                    int anchor_count, float node_coverage, bool threshold_nodes_by_distance, int minimum_valid_anchor_count) {
	auto device = input_mesh.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeData(device, nodes, node_rotations, node_translations);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	// endregion

	TriangleMesh warped_mesh(device);

	if (input_mesh.HasTriangles()) {
		warped_mesh.SetTriangles(input_mesh.GetTriangles());
	}
	if (input_mesh.HasVertexColors()) {
		warped_mesh.SetVertexColors(input_mesh.GetVertexColors());
	}
	if (input_mesh.HasTriangleColors()) {
		warped_mesh.SetTriangleColors(input_mesh.GetTriangleColors());
	}

	if (input_mesh.HasVertices()) {
		const auto& vertices = input_mesh.GetVertices();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		vertices.AssertDtype(core::Dtype::Float32);
		core::Tensor warped_vertices;
		if (threshold_nodes_by_distance) {
			kernel::warp::WarpPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
			                         minimum_valid_anchor_count);
		} else {
			kernel::warp::WarpPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage);
		}

		warped_mesh.SetVertices(warped_vertices);
	}


	return warped_mesh;
}

void ComputeAnchorsAndWeightsEuclidean(core::Tensor& anchors, core::Tensor& weights, const core::Tensor& points, const core::Tensor& nodes,
                                       int anchor_count, int minimum_valid_anchor_count, float node_coverage) {
	auto device = points.GetDevice();
	points.AssertDtype(core::Dtype::Float32);
	nodes.AssertDtype(core::Dtype::Float32);
	nodes.AssertDevice(device);
	if (minimum_valid_anchor_count > anchor_count) {
		utility::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
		                  minimum_valid_anchor_count, anchor_count);
	}
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	kernel::graph::ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
}

py::tuple ComputeAnchorsAndWeightsEuclidean(const core::Tensor& points, const core::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
                                            float node_coverage) {
	core::Tensor anchors, weights;
	ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
	return py::make_tuple(anchors, weights);
}

void ComputeAnchorsAndWeightsShortestPath(core::Tensor& anchors, core::Tensor& weights, const core::Tensor& points, const core::Tensor& nodes,
                                          const core::Tensor& edges, int anchor_count, float node_coverage) {
	auto device = points.GetDevice();
	nodes.AssertDevice(device);
	edges.AssertDevice(device);
	points.AssertDtype(core::Dtype::Float32);
	nodes.AssertDtype(core::Dtype::Float32);
	edges.AssertDtype(core::Dtype::Int32);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	kernel::graph::ComputeAnchorsAndWeightsShortestPath(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
}

py::tuple ComputeAnchorsAndWeightsShortestPath(const core::Tensor& points, const core::Tensor& nodes, const core::Tensor& edges, int anchor_count,
                                               float node_coverage) {
	core::Tensor anchors, weights;
	ComputeAnchorsAndWeightsShortestPath(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
	return py::make_tuple(anchors, weights);
}

} // namespace nnrt::geometry