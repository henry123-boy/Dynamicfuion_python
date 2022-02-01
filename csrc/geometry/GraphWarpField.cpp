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
#include "geometry/GraphWarpField.h"

#include <utility>
#include <Eigen/Core>
#include "geometry/kernel/Graph.h"
#include "geometry/kernel/Warp.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3tg = open3d::t::geometry;


namespace nnrt::geometry {

void CheckNodeMatrixTransformationData(o3c::Device& device, const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
                                       const o3c::Tensor& node_translations) {
	if (device != nodes.GetDevice() || device != node_rotations.GetDevice() || device != node_translations.GetDevice()) {
		o3u::LogError("Device not consistent among arguments.");
	}
	auto nodes_shape = nodes.GetShape();
	auto rotations_shape = node_rotations.GetShape();
	auto translations_shape = node_translations.GetShape();
	if (nodes_shape.size() != 2 || rotations_shape.size() != 3 || translations_shape.size() != 2) {
		o3u::LogError("Arguments nodes, rotations, and translations need to have 2, 3, and 2 dimensions,"
		              " respectively. Got {}, {}, and {}.", nodes_shape.size(),
		              rotations_shape.size(), translations_shape.size());
	}

	const int64_t node_count = nodes_shape[0];
	if (nodes_shape[1] != 3) {
		o3u::LogError("Argument nodes needs to have size N x 3, has size N x {}.", nodes_shape[1]);
	}
	if (rotations_shape[0] != node_count || rotations_shape[1] != 3 || rotations_shape[2] != 3) {
		o3u::LogError("Argument node_rotations needs to have shape ({}, 3, 3), where first dimension is the node count N"
		              ", but has shape {}", node_count, rotations_shape);
	}
	if (translations_shape[0] != node_count || translations_shape[1] != 3) {
		o3u::LogError("Argument node_translations needs to have shape ({}, 3), where first dimension is the node count N"
		              ", but has shape {}", node_count, translations_shape);
	}

	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(node_rotations, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(node_translations, o3c::Dtype::Float32);
}

o3tg::PointCloud
WarpPointCloud(const o3tg::PointCloud& input_point_cloud, const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
               const o3c::Tensor& node_translations, int anchor_count, float node_coverage,
               int minimum_valid_anchor_count) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
	if (anchor_count < 1) {
		o3u::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	if (anchor_count < 0 || anchor_count > MAX_ANCHOR_COUNT) {
		o3u::LogError("`anchor_count` is {}, but is required to satisfy 0 < anchor_count <= {}", anchor_count, MAX_ANCHOR_COUNT);
	}
	if (minimum_valid_anchor_count < 0 || minimum_valid_anchor_count > anchor_count) {
		o3u::LogError("`minimum_valid_anchor_count` is {}, but is required to satisfy 0 < minimum_valid_anchor_count <= {} ",
		              minimum_valid_anchor_count, anchor_count);
	}
	// endregion

	o3tg::PointCloud warped_point_cloud(device);

	if (warped_point_cloud.HasPointColors()) {
		warped_point_cloud.SetPointColors(input_point_cloud.GetPointColors());
	}


	if (input_point_cloud.HasPointPositions()) {
		const auto& vertices = input_point_cloud.GetPointPositions();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		o3c::AssertTensorDtype(vertices, o3c::Dtype::Float32);
		o3c::Tensor warped_points;
		kernel::warp::WarpPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
		                         minimum_valid_anchor_count);
		warped_point_cloud.SetPointPositions(warped_points);
	}

	return warped_point_cloud;
}


o3tg::PointCloud
WarpPointCloud(const o3tg::PointCloud& input_point_cloud, const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
               const o3c::Tensor& node_translations, const o3c::Tensor& anchors, const o3c::Tensor& anchor_weights,
               int minimum_valid_anchor_count) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
	auto anchors_shape = anchors.GetShape();
	auto anchor_weights_shape = anchor_weights.GetShape();
	if (anchors_shape.size() != 2 || anchor_weights_shape.size() != 2) {
		o3u::LogError("Tensors `anchors` and `anchor_weights` need to both have two dimensions."
		              "Got {} and {} dimensions, respectively.", anchors_shape.size(),
		              anchor_weights_shape.size());
	}
	if (anchors_shape[0] != anchor_weights_shape[0] || anchors_shape[1] != anchor_weights_shape[1]) {
		o3u::LogError("Tensors `anchors` and `anchor_weights` need to have matching dimensions."
		              "Got {} and {}, respectively.", anchors_shape,
		              anchor_weights_shape);
	}
	const int64_t anchor_count = anchors_shape[1];
	if (minimum_valid_anchor_count < 0 || minimum_valid_anchor_count > anchor_count) {
		o3u::LogError("`minimum_valid_anchor_count` is {}, but is required to satisfy 0 < minimum_valid_anchor_count <= {}, "
		              "where the upper bound is the second dimension of the input `anchors` tensor.",
		              minimum_valid_anchor_count, anchor_count);
	}
	o3c::AssertTensorDtype(anchors, o3c::Dtype::Int32);
	o3c::AssertTensorDtype(anchor_weights, o3c::Dtype::Float32);

	o3tg::PointCloud warped_point_cloud(device);

	if (warped_point_cloud.HasPointColors()) {
		warped_point_cloud.SetPointColors(input_point_cloud.GetPointColors());
	}


	if (input_point_cloud.HasPointPositions()) {
		const auto& vertices = input_point_cloud.GetPointPositions();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		o3c::AssertTensorDtype(vertices, o3c::Dtype::Float32);
		o3c::Tensor warped_points;
		kernel::warp::WarpPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchors, anchor_weights,
		                         minimum_valid_anchor_count);
		warped_point_cloud.SetPointPositions(warped_points);
	}

	return warped_point_cloud;
}

inline
void CopyWarpedTriangleMeshData(o3tg::TriangleMesh& warped_mesh, const o3tg::TriangleMesh& input_mesh) {
	if (input_mesh.HasTriangleIndices()) {
		warped_mesh.SetTriangleIndices(input_mesh.GetTriangleIndices());
	}
	if (input_mesh.HasVertexColors()) {
		warped_mesh.SetVertexColors(input_mesh.GetVertexColors());
	}
	if (input_mesh.HasTriangleColors()) {
		warped_mesh.SetTriangleColors(input_mesh.GetTriangleColors());
	}
}

o3tg::TriangleMesh
WarpTriangleMesh(const o3tg::TriangleMesh& input_mesh, const o3c::Tensor& nodes,
                 const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
                 int anchor_count, float node_coverage, bool threshold_nodes_by_distance, int minimum_valid_anchor_count) {
	auto device = input_mesh.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
	if (anchor_count < 1) {
		o3u::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	// endregion

	o3tg::TriangleMesh warped_mesh(device);

	CopyWarpedTriangleMeshData(warped_mesh, input_mesh);

	if (input_mesh.HasVertexPositions()) {
		const auto& vertices = input_mesh.GetVertexPositions();
		// FIXME: not sure if this check is at all necessary. There seem to be some situations in pythonic context when np.array(mesh.vertices)
		//  materializes in np.float64 datatype, e.g. after generation of a box using standard API functions. This was true for Open3D 0.12.0.
		o3c::AssertTensorDtype(vertices, o3c::Dtype::Float32);
		o3c::Tensor warped_vertices;
		if (threshold_nodes_by_distance) {
			kernel::warp::WarpPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
			                         minimum_valid_anchor_count);
		} else {
			kernel::warp::WarpPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage);
		}

		warped_mesh.SetVertexPositions(warped_vertices);
	}


	return warped_mesh;
}

void ComputeAnchorsAndWeightsEuclidean(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
                                       int anchor_count, int minimum_valid_anchor_count, float node_coverage) {
	auto device = points.GetDevice();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDevice(nodes, device);
	if (minimum_valid_anchor_count > anchor_count) {
		o3u::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
		              minimum_valid_anchor_count, anchor_count);
	}
	if (anchor_count < 1) {
		o3u::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	kernel::graph::ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
}

py::tuple ComputeAnchorsAndWeightsEuclidean(const o3c::Tensor& points, const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
                                            float node_coverage) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
	return py::make_tuple(anchors, weights);
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
		o3u::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
	}
	kernel::graph::ComputeAnchorsAndWeightsShortestPath(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
}

py::tuple ComputeAnchorsAndWeightsShortestPath(const o3c::Tensor& points, const o3c::Tensor& nodes, const o3c::Tensor& edges, int anchor_count,
                                               float node_coverage) {
	o3c::Tensor anchors, weights;
	ComputeAnchorsAndWeightsShortestPath(anchors, weights, points, nodes, edges, anchor_count, node_coverage);
	return py::make_tuple(anchors, weights);
}

GraphWarpField::GraphWarpField(o3c::Tensor nodes, o3c::Tensor edges, o3c::Tensor edge_weights,
                               o3c::Tensor clusters, float node_coverage, bool threshold_nodes_by_distance, int anchor_count,
                               int minimum_valid_anchor_count) :
		nodes(std::move(nodes)), edges(std::move(edges)), edge_weights(std::move(edge_weights)), clusters(std::move(clusters)),
		translations({this->nodes.GetLength(), 3}, o3c::Dtype::Float32, this->nodes.GetDevice()),
		rotations({this->nodes.GetLength(), 3, 3}, o3c::Dtype::Float32, this->nodes.GetDevice()),
		node_coverage(node_coverage), threshold_nodes_by_distance(threshold_nodes_by_distance), anchor_count(anchor_count),
		minimum_valid_anchor_count(minimum_valid_anchor_count) {
	auto device = this->nodes.GetDevice();
	o3c::AssertTensorDevice(this->edges, device);
	o3c::AssertTensorDevice(this->edge_weights, device);
	o3c::AssertTensorDevice(this->clusters, device);
	auto nodes_shape = this->nodes.GetShape();
	auto edges_shape = this->edges.GetShape();
	auto edge_weights_shape = this->edge_weights.GetShape();
	auto clusters_shape = this->clusters.GetShape();
	if (nodes_shape.size() != 2 || edges_shape.size() != 2 || edge_weights_shape.size() != 2 || clusters_shape.size() != 1) {
		o3u::LogError("Arguments `nodes`, `edges`, and `edge_weights` all need to have two dimensions,"
		              " while `clusters` needs to have one dimension. Got dimension counts {}, {}, {}, and {}, respectively.",
		              nodes_shape.size(), edges_shape.size(), edge_weights_shape.size(), clusters_shape.size());
	}
	const int64_t node_count = nodes_shape[0];
	if (nodes_shape[1] != 3) {
		o3u::LogError("Argument nodes needs to have size N x 3, has size N x {}.", nodes_shape[1]);
	}
	if (edges_shape[0] != node_count) {
		o3u::LogError("Argument `edges_shape` needs to have shape ({}, X), where first dimension is the node count N"
		              " and the second is the edge degree X, but has shape {}", node_count, edges_shape);
	}
	if (edge_weights_shape != edges_shape) {
		o3u::LogError("arguments `edges` & `edge_weights` need to have the same shape. Got shapes: {} and {}, respectively.", edges_shape,
		              edge_weights_shape);
	}
	if (clusters_shape[0] != node_count) {
		o3u::LogError("argument `clusters` needs to be a vector of the size {} (node count), got size {}.", node_count, clusters_shape[0]);
	}
}

o3c::Tensor GraphWarpField::GetWarpedNodes() const {
	return nodes + this->translations;
}

o3c::TensorList GraphWarpField::GetNodeExtent() const {
	auto max = nodes.Max({0});
	auto min = nodes.Min({0});
	return o3c::TensorList({min, max});
}

open3d::t::geometry::TriangleMesh
GraphWarpField::WarpMesh(const open3d::t::geometry::TriangleMesh& input_mesh, bool disable_neighbor_thresholding) const {
	if (disable_neighbor_thresholding) {
		return WarpTriangleMesh(input_mesh, this->nodes, this->rotations, this->translations, this->anchor_count, this->node_coverage,
		                        false, 0);
	} else {
		return WarpTriangleMesh(input_mesh, this->nodes, this->rotations, this->translations, this->anchor_count, this->node_coverage,
		                        this->threshold_nodes_by_distance, this->minimum_valid_anchor_count);
	}

}


} // namespace nnrt::geometry