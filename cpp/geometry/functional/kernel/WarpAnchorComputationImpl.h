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

#include "core/platform_independence/Qualifiers.h"
#include "WarpUtilities.h"
#include "geometry/functional/kernel/WarpAnchorComputation.h"


using namespace open3d;
namespace o3c = open3d::core;
namespace utility = open3d::utility;
using namespace open3d::t::geometry::kernel;


namespace nnrt::geometry::functional::kernel {

// region ======================================== EUCLIDEAN =========================================================================================
template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
void ComputeAnchorsAndWeights_Euclidean_VariableNodeWeight(
		open3d::core::Tensor& anchors,
		open3d::core::Tensor& anchor_weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		int anchor_count,
		int minimum_valid_anchor_count
) {
	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	anchor_weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());
	if(!node_coverage_weights.IsContiguous()){
		utility::LogError("node_coverage_weights is not contiguous.");
	}

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	auto node_coverage_weight_data = node_coverage_weights.GetDataPtr<float>();

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(anchor_weights, 1);
	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_point) {
				auto point_data = point_indexer.GetDataPtr<float>(i_point);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(i_point);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(i_point);
				if (TUseValidAnchorThreshold) {
					warp::FindAnchorsAndWeightsForPoint_Euclidean_Threshold_VariableNodeCoverageWeight<TDeviceType>(
							anchor_indices, anchor_weights, anchor_count,
							minimum_valid_anchor_count, node_count,
							point, node_indexer, node_coverage_weight_data
					);
				} else {
					warp::FindAnchorsAndWeightsForPoint_Euclidean_VariableNodeCoverageWeight<TDeviceType>(
							anchor_indices, anchor_weights,
							anchor_count, node_count,
							point, node_indexer, node_coverage_weight_data
					);
				}
				//endregion
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
void ComputeAnchorsAndWeights_Euclidean_FixedNodeWeight
		(
				open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
				const open3d::core::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
				float node_coverage
		) {

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
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);
				if (TUseValidAnchorThreshold) {
					warp::FindAnchorsAndWeightsForPoint_Euclidean_Threshold_FixedNodeCoverageWeight<TDeviceType>(anchor_indices, anchor_weights,
					                                                                                             anchor_count,
					                                                                                             minimum_valid_anchor_count,
					                                                                                             node_count,
					                                                                                             point, node_indexer,
					                                                                                             node_coverage_squared);
				} else {
					warp::FindAnchorsAndWeightsForPoint_Euclidean_FixedNodeCoverageWeight<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
					                                                                                   node_count,
					                                                                                   point, node_indexer, node_coverage_squared);
				}
				// endregion
			}
	);
}

// endregion
// region ======================================== SHORTEST PATH =====================================================================================
template<o3c::Device::DeviceType TDeviceType>
void ComputeAnchorsAndWeights_ShortestPath_VariableNodeWeight(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights,

		const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& edges,
		int anchor_count
) {
	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer edge_indexer(edges, 1);
	auto node_coverage_weight_data = node_coverage_weights.GetDataPtr<float>();

	o3c::AssertTensorShape(edges, { node_count, utility::nullopt });
	int graph_degree = static_cast<int>(edges.GetShape()[1]);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);

				warp::FindAnchorsAndWeightsForPoint_ShortestPath_VariableNodeCoverageWeight<TDeviceType>(
						anchor_indices, anchor_weights, anchor_count, node_count,
						point, node_indexer, edge_indexer,
						node_coverage_weight_data,
						graph_degree
				);
				// endregion
			}
	);
}

template<o3c::Device::DeviceType TDeviceType>
void ComputeAnchorsAndWeights_ShortestPath_FixedNodeWeight(
		open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points, const open3d::core::Tensor& nodes,
		const open3d::core::Tensor& edges, int anchor_count, float node_coverage
) {

	float node_coverage_squared = node_coverage * node_coverage;
	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);
	NDArrayIndexer edge_indexer(edges, 1);

	o3c::AssertTensorShape(edges, { node_count, utility::nullopt });
	int graph_degree = static_cast<int>(edges.GetShape()[1]);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

	open3d::core::ParallelFor(
			points.GetDevice(), point_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);

				warp::FindAnchorsAndWeightsForPoint_ShortestPath_FixedNodeCoverageWeight<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
				                                                                                      node_count,
				                                                                                      point, node_indexer, edge_indexer,
				                                                                                      node_coverage_squared,
				                                                                                      graph_degree);
				// endregion
			}
	);
}
// endregion
} // namespace nnrt::geometry::functional::kernel