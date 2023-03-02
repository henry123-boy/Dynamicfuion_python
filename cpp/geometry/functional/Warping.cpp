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
// local
#include "Warping.h"
#include "geometry/functional/kernel/Warp3dPointsAndNormals.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry::functional {


void CheckNodeMatrixTransformationData(
		o3c::Device& device, const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
		const o3c::Tensor& node_translations
) {
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

	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(node_rotations, o3c::Dtype::Float32);
	o3c::AssertTensorDtype(node_translations, o3c::Dtype::Float32);
}

o3tg::PointCloud WarpPointCloud(
		const o3tg::PointCloud& input_point_cloud, const o3c::Tensor& nodes, const o3c::Tensor& node_rotations,
		const o3c::Tensor& node_translations, int anchor_count, float node_coverage,
		int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics /*= open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
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
		kernel::warp::Warp3dPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
		                           minimum_valid_anchor_count, extrinsics);
		warped_point_cloud.SetPointPositions(warped_points);
	}

	return warped_point_cloud;
}


o3tg::PointCloud WarpPointCloud(
		const o3tg::PointCloud& input_point_cloud,
		const o3c::Tensor& nodes, const o3c::Tensor& node_rotations, const o3c::Tensor& node_translations,
		const o3c::Tensor& anchors, const o3c::Tensor& anchor_weights,
		int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics /*= open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))*/
) {
	auto device = input_point_cloud.GetDevice();
	// region ================ INPUT CHECKS ======================================
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
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
		kernel::warp::Warp3dPoints(warped_points, vertices, nodes, node_rotations, node_translations, anchors, anchor_weights,
		                           minimum_valid_anchor_count, extrinsics);
		warped_point_cloud.SetPointPositions(warped_points);
	}

	return warped_point_cloud;
}

inline
void CopyTransformIndependentTriangleMeshData(o3tg::TriangleMesh& output_mesh, const o3tg::TriangleMesh& input_mesh) {
	if (input_mesh.HasTriangleIndices()) {
		output_mesh.SetTriangleIndices(input_mesh.GetTriangleIndices());
	}
	if (input_mesh.HasVertexColors()) {
		output_mesh.SetVertexColors(input_mesh.GetVertexColors());
	}
	if (input_mesh.HasTriangleColors()) {
		output_mesh.SetTriangleColors(input_mesh.GetTriangleColors());
	}
}

o3tg::TriangleMesh WarpTriangleMesh(
		const o3tg::TriangleMesh& input_mesh,
		const o3c::Tensor& nodes,
		const o3c::Tensor& node_rotations,
		const o3c::Tensor& node_translations,
		int anchor_count,
		float node_coverage,
		bool threshold_nodes_by_distance,
		int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics /*= open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")*/
) {
	auto device = input_mesh.GetDevice();
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
	if (anchor_count < 1) {
		utility::LogError("anchor_count needs to be at least than one. Got: {}.", anchor_count);
	}

	o3tg::TriangleMesh warped_mesh(device);

	CopyTransformIndependentTriangleMeshData(warped_mesh, input_mesh);

	if (input_mesh.HasVertexPositions()) {
		const auto& vertices = input_mesh.GetVertexPositions();
		o3c::AssertTensorDtype(vertices, o3c::Dtype::Float32);
		o3c::Tensor warped_vertices;
		if (input_mesh.HasVertexNormals()) {
			const auto& vertex_normals = input_mesh.GetVertexNormals();
			o3c::AssertTensorDtype(vertex_normals, o3c::Dtype::Float32);
			o3c::Tensor warped_normals;
			if (threshold_nodes_by_distance) {
				kernel::warp::Warp3dPointsAndNormals(warped_vertices, warped_normals, vertices, vertex_normals, nodes, node_rotations,
				                                     node_translations, anchor_count, node_coverage, minimum_valid_anchor_count, extrinsics);
			} else {
				kernel::warp::Warp3dPointsAndNormals(warped_vertices, warped_normals, vertices, vertex_normals, nodes, node_rotations,
				                                     node_translations, anchor_count, node_coverage, extrinsics);
			}
			warped_mesh.SetVertexNormals(warped_normals);
		} else {
			if (threshold_nodes_by_distance) {
				kernel::warp::Warp3dPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
				                           minimum_valid_anchor_count, extrinsics);
			} else {
				kernel::warp::Warp3dPoints(warped_vertices, vertices, nodes, node_rotations, node_translations, anchor_count, node_coverage,
				                           extrinsics);
			}
		}

		warped_mesh.SetVertexPositions(warped_vertices);
	}

	return warped_mesh;
}

open3d::t::geometry::TriangleMesh WarpTriangleMeshUsingSuppliedAnchors(
		const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& node_translations, const open3d::core::Tensor& anchors,
		const open3d::core::Tensor& anchor_weights, bool threshold_nodes_by_distance,
		int minimum_valid_anchor_count, const open3d::core::Tensor& extrinsics
) {
	auto device = input_mesh.GetDevice();
	CheckNodeMatrixTransformationData(device, nodes, node_rotations, node_translations);
	o3tg::TriangleMesh warped_mesh(device);
	CopyTransformIndependentTriangleMeshData(warped_mesh, input_mesh);

	if (input_mesh.HasVertexPositions()) {
		const auto& vertices = input_mesh.GetVertexPositions();
		o3c::AssertTensorDtype(vertices, o3c::Dtype::Float32);
		o3c::Tensor warped_vertices;
		if (input_mesh.HasVertexNormals()) {
			const auto& vertex_normals = input_mesh.GetVertexNormals();
			o3c::AssertTensorDtype(vertex_normals, o3c::Dtype::Float32);
			o3c::Tensor warped_normals;
			if (threshold_nodes_by_distance) {
				kernel::warp::Warp3dPointsAndNormals(warped_vertices, warped_normals, vertices, vertex_normals, nodes, node_rotations,
				                                     node_translations, anchors, anchor_weights, minimum_valid_anchor_count, extrinsics);
			} else {
				kernel::warp::Warp3dPointsAndNormals(warped_vertices, warped_normals, vertices, vertex_normals, nodes, node_rotations,
				                                     node_translations, anchors, anchor_weights, extrinsics);
			}
			warped_mesh.SetVertexNormals(warped_normals);
		} else {
			if (threshold_nodes_by_distance) {
				kernel::warp::Warp3dPoints(warped_vertices, vertices, nodes, node_rotations,
				                           node_translations, anchors, anchor_weights, minimum_valid_anchor_count, extrinsics);
			} else {
				kernel::warp::Warp3dPoints(warped_vertices, vertices, nodes, node_rotations,
				                           node_translations, anchors, anchor_weights, extrinsics);
			}
		}

		warped_mesh.SetVertexPositions(warped_vertices);
	}

	return warped_mesh;
}

} // namespace nnrt::geometry::functional