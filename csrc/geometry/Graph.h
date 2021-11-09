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
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nnrt::geometry {

open3d::t::geometry::PointCloud
WarpPointCloudMat(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
                  const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  int anchor_count, float node_coverage,
                  int minimum_valid_anchor_count = 0);

open3d::t::geometry::PointCloud
WarpPointCloudMat(const open3d::t::geometry::PointCloud& input_point_cloud, const open3d::core::Tensor& nodes,
                  const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
                  int minimum_valid_anchor_count = 0);

open3d::t::geometry::TriangleMesh WarpTriangleMeshMat(const open3d::t::geometry::TriangleMesh& input_mesh, const open3d::core::Tensor& nodes,
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


} // namespace nnrt::geometry