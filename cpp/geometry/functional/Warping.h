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
#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/geometry/TriangleMesh.h>

namespace nnrt::geometry::functional{

// computes anchors on-the-fly, uses node thresholding
open3d::t::geometry::PointCloud WarpPointCloud(
		const open3d::t::geometry::PointCloud& input_point_cloud,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		int minimum_valid_anchor_count = 0,
		const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))
);

// uses precomputed anchors, uses node thresholding
open3d::t::geometry::PointCloud WarpPointCloud(
		const open3d::t::geometry::PointCloud& input_point_cloud,
		const open3d::core::Tensor& nodes,const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		int minimum_valid_anchor_count = 0,
		const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))
);

// computes anchors on-the-fly
open3d::t::geometry::TriangleMesh WarpTriangleMesh(
		const open3d::t::geometry::TriangleMesh& input_mesh,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		bool threshold_nodes_by_distance = true, int minimum_valid_anchor_count = 0,
		const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))
);

// uses precomputed anchors
open3d::t::geometry::TriangleMesh WarpTriangleMeshUsingSuppliedAnchors(
		const open3d::t::geometry::TriangleMesh& input_mesh,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		bool threshold_nodes_by_distance = true, int minimum_valid_anchor_count = 0,
		const open3d::core::Tensor& extrinsics = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0"))
);

} // namespace nnrt::geometry::functional