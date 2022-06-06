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

#include <cmath>
#include <Eigen/Dense>

#include <open3d/core/Tensor.h>
#include <open3d/core/TensorKey.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/utility/Console.h>

#include "core/PlatformIndependence.h"
#include "geometry/kernel/WarpUtilities.h"
#include "Graph.h"


using namespace open3d;
namespace o3c = open3d::core;
namespace o3u = open3d::utility;
using namespace open3d::t::geometry::kernel;



namespace nnrt::geometry::kernel::graph {

template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
void ComputeAnchorsAndWeightsEuclidean
		(open3d::core::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points,
		 const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
		 const float node_coverage) {

	float node_coverage_squared = node_coverage * node_coverage;

	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);
				if (TUseValidAnchorThreshold) {
					warp::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
					                                                                     minimum_valid_anchor_count, node_count,
					                                                                     point, node_indexer, node_coverage_squared);
				} else {
					warp::FindAnchorsAndWeightsForPointEuclidean<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
					                                                           point, node_indexer, node_coverage_squared);
				}
				// endregion
			}
	);
}
template<o3c::Device::DeviceType TDeviceType>
void ComputeAnchorsAndWeightsShortestPath(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
                                          const o3c::Tensor& edges, int anchor_count, float node_coverage) {

	float node_coverage_squared = node_coverage * node_coverage;
	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer edge_indexer(edges, 1);

	o3c::AssertTensorShape(edges, {node_count, o3u::nullopt});
	int graph_degree = static_cast<int>(edges.GetShape()[1]);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);

				warp::FindAnchorsAndWeightsForPointShortestPath<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
				                                                             point, node_indexer, edge_indexer, node_coverage_squared,
																			 graph_degree);
				// endregion
			}
	);
}

} // namespace nnrt::geometry::kernel::graph