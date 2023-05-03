//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/25/22.
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
//TODO: remove header / replace implementation if unused
//stdlib
#include <cmath>
#include <cfloat>

// 3rd party
#include <open3d/core/hashmap/HashMap.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

// local
#include "GeometrySampling.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/platform_independence/Atomics.h"
#include "core/kernel/HashTableUtilities.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/functional/ParallelPrefixScan.h"
#include "geometry/functional/kernel/PointAggregationBins.h"
#include "geometry/functional/kernel/GeometrySamplingMean.h"
#include "geometry/functional/kernel/GeometrySamplingMedian.h"


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;
namespace hash_table = nnrt::core::kernel::hash_table;
namespace utility = open3d::utility;

namespace nnrt::geometry::functional::kernel::sampling {


template<open3d::core::Device::DeviceType TDeviceType>
void GridDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend
) {
	mean::ComputeMeanOfPointsInGridCells<TDeviceType>(downsampled_points, original_points, grid_cell_size, hash_backend);
}

template<open3d::core::Device::DeviceType TDeviceType>
void FastRadiusDownsamplePoints(
		open3d::core::Tensor& downsampled_points, const open3d::core::Tensor& original_points, float radius,
		const open3d::core::HashBackendType& hash_backend
) {
	o3c::Tensor downsampled_points_stage_1;
	float extended_radius = sqrtf(2 * (radius * radius));
	mean::ComputeMeanOfPointsInGridCells<TDeviceType>(downsampled_points_stage_1, original_points, extended_radius * 2, hash_backend);
	// merge again while offsetting the grid
	mean::ComputeMeanOfPointsInGridCells<TDeviceType>(downsampled_points, downsampled_points_stage_1, extended_radius * 2, hash_backend,
	                                                  Eigen::Vector3f(0.5, 0.5, 0.5));
}


template<open3d::core::Device::DeviceType TDeviceType>
void GridMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend_type
) {
	median::MedianGridSamplePoints<TDeviceType>(sample, points, grid_cell_size, hash_backend_type);
}

template<open3d::core::Device::DeviceType TDeviceType>
void RadiusMedianSubsample3dPoints(
		open3d::core::Tensor& sample,
		const open3d::core::Tensor& points,
		float radius,
		const open3d::core::HashBackendType& hash_backend_type
) {
	// TODO: need to reject >radius points from cluster instead
	// float extended_radius = sqrtf(2 * (radius * radius));
	float extended_radius = radius;
	o3c::Tensor median_point_indices_stage1, median_point_indices_stage2;
	median::MedianGridSamplePoints<TDeviceType>(median_point_indices_stage1, points, extended_radius * 2, hash_backend_type);
	o3c::Tensor points_stage_1 = points.GetItem(o3c::TensorKey::IndexTensor(median_point_indices_stage1));
	// merge again while offsetting the grid
	median::MedianGridSamplePoints<TDeviceType>(median_point_indices_stage2, points_stage_1, extended_radius * 2, hash_backend_type,
	                                            Eigen::Vector3f(0.5, 0.5, 0.5));
	sample = median_point_indices_stage1.GetItem(o3c::TensorKey::IndexTensor(median_point_indices_stage2));
}


template<open3d::core::Device::DeviceType TDeviceType>
void RadiusSubsampleGraph(
		open3d::core::Tensor& resampled_vertices,
		open3d::core::Tensor& resampled_edges,
		const open3d::core::Tensor& vertices,
		const open3d::core::Tensor& edges,
		float radius
) {
	// counters and checks
	o3c::Device device = vertices.GetDevice();
	int64_t vertex_count = vertices.GetLength();
	o3c::AssertTensorShape(vertices, { vertex_count, 3 });
	o3c::AssertTensorDtype(vertices, o3c::Float32);

	int64_t max_vertex_degree = edges.GetShape(1);
	o3c::AssertTensorShape(edges, { vertex_count, max_vertex_degree });
	o3c::AssertTensorDtype(edges, o3c::Int64);
	o3c::AssertTensorDevice(edges, device);

	if (radius < 1e-7) {
		utility::LogError("Radius must be a positive value above 1e-7. Provided radius: {}", radius);
	}

	// prep inputs
	auto* vertex_data = vertices.GetDataPtr<float>();
	auto* edge_data = edges.GetDataPtr<int64_t>();

	// init output for this phase

	// find super-independent vertices
	o3c::Tensor super_independent_mask_tier1 = o3c::Tensor::Zeros({vertex_count}, o3c::Bool, device);
	auto* super_independent_mask1_data = super_independent_mask_tier1.GetDataPtr<bool>();
	NNRT_DECLARE_ATOMIC(int, super_independent_count);
	o3c::Tensor super_independent_set = o3c::Tensor({vertex_count}, o3c::Int64, device);
	auto super_independent_set_data = super_independent_set.GetDataPtr<int64_t>();

	// greedy algorithm, prone to race conditions
	o3c::ParallelFor(
			device, vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_source_vertex) {
				if (super_independent_mask1_data[i_source_vertex]) {
					return;
				} else {
					auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
					for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
						int64_t i_target_vertex = vertex_edges[i_edge];
						if (i_target_vertex == -1) {
							break;
						} else if (super_independent_mask1_data[i_target_vertex]) {
							// neighbor already in set, this source should be excluded
							return;
						}
					}
					super_independent_mask1_data[i_source_vertex] = true;
					int index = NNRT_ATOMIC_ADD(super_independent_count, 1);
					super_independent_set_data[index] = i_source_vertex;
				}
			}
	);

	// check for race conditions: remove any extra vertices from the set (which have distance < 3 edges from another vertex in the set)
	o3c::Tensor super_independent_mask_tier2 = o3c::Tensor::Ones({NNRT_GET_ATOMIC_VALUE_HOST(super_independent_count)}, o3c::Bool, device);
	auto* super_independent_mask2_data = super_independent_mask_tier2.GetDataPtr<bool>();
	o3c::ParallelFor(
			device, NNRT_GET_ATOMIC_VALUE_HOST(super_independent_count),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_super_independent_vertex) {
				int64_t i_source_vertex = super_independent_set_data[i_super_independent_vertex];
				auto vertex_edges_1_ring = edge_data + i_source_vertex * max_vertex_degree;
				for (int i_edge_1_ring = 0; i_edge_1_ring < max_vertex_degree; i_edge_1_ring++) {
					int64_t i_target_vertex_1_ring = vertex_edges_1_ring[i_edge_1_ring];
					if (i_target_vertex_1_ring == -1) {
						break;
					} else if (super_independent_mask1_data[i_target_vertex_1_ring] && super_independent_mask2_data[i_target_vertex_1_ring]) {
						// distance is only 1 edge
						super_independent_mask2_data[i_super_independent_vertex] = false;
					} else {
						auto vertex_edges_2_ring = edge_data + i_target_vertex_1_ring * max_vertex_degree;
						for (int i_edge_2_ring = 0; i_edge_2_ring < max_vertex_degree; i_edge_2_ring++) {
							int64_t i_target_vertex_2_ring = vertex_edges_2_ring[i_edge_2_ring];
							if (i_target_vertex_2_ring == -1) {
								break;
							} else if (super_independent_mask1_data[i_target_vertex_2_ring] && super_independent_mask2_data[i_target_vertex_2_ring]) {
								// distance is only 2 edges
								super_independent_mask2_data[i_super_independent_vertex] = false;
							}
						}
					}
				}
			}
	);

	super_independent_set = super_independent_set.GetItem(o3c::TensorKey::IndexTensor(super_independent_mask_tier2));NNRT_CLEAN_UP_ATOMIC(
			super_independent_count);

	int64_t super_independent_vertex_count = super_independent_set.GetLength();
	super_independent_set_data = super_independent_set.GetDataPtr<int64_t>();
	o3c::Tensor new_vertex_indices({vertex_count}, o3c::Int64, device);
	new_vertex_indices.Fill(-1);
	auto new_vertex_index_data = new_vertex_indices.GetDataPtr<int64_t>();
	o3c::Tensor new_vertex_positions({vertex_count, 3}, o3c::Float32, device);
	auto new_vertex_position_data = new_vertex_positions.GetDataPtr<float>();

	NNRT_DECLARE_ATOMIC(int, new_vertex_count);

	// collapse edges around super-independent vertices if they are shorter than radius
	o3c::ParallelFor(
			device, super_independent_vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_super_independent_vertex) {
				int64_t i_source_vertex = super_independent_set_data[i_super_independent_vertex];
				Eigen::Map<const Eigen::Vector3f> source_vertex(vertex_data + i_source_vertex * 3);

				int new_merged_vertex_index = NNRT_ATOMIC_ADD(new_vertex_count, 1);
				new_vertex_index_data[i_source_vertex] = new_merged_vertex_index;

				Eigen::Map<Eigen::Vector3f> averaged_vertex(new_vertex_position_data + new_merged_vertex_index * 3);
				averaged_vertex = Eigen::Vector3f(0.f, 0.f, 0.f);
				averaged_vertex += source_vertex;
				int averaged_vertex_count = 0;

				auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
				for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
					int64_t i_target_vertex = vertex_edges[i_edge];
					Eigen::Map<const Eigen::Vector3f> target_vertex(vertex_data + i_target_vertex * 3);
					if ((source_vertex - target_vertex).norm() < radius) {
						averaged_vertex_count++;
						averaged_vertex += target_vertex;
						new_vertex_index_data[i_target_vertex] = new_merged_vertex_index;
					} else {
						int unmerged_vertex_index = NNRT_ATOMIC_ADD(new_vertex_count, 1);
						new_vertex_index_data[i_target_vertex] = unmerged_vertex_index;
						memcpy(new_vertex_position_data + unmerged_vertex_index * 3, vertex_data + i_target_vertex * 3, 3 * sizeof(float));
					}
				}
				averaged_vertex /= static_cast<float>(averaged_vertex_count);
			}
	);

	int64_t new_max_vertex_degree = max_vertex_degree * 4;

	resampled_vertices = new_vertex_positions.Slice(0, 0, NNRT_GET_ATOMIC_VALUE_HOST(new_vertex_count));NNRT_CLEAN_UP_ATOMIC(new_vertex_count);

	resampled_edges = o3c::Tensor({vertices.GetLength(), new_max_vertex_degree}, o3c::Int64, device);
	resampled_edges.Fill(-1);
	auto new_edge_data = resampled_edges.GetDataPtr<int64_t>();
	core::AtomicCounterArray<TDeviceType> new_edge_counts(vertices.GetLength());

	// reconstruct edges from new indices
	o3c::ParallelFor(
			device, vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_source_vertex) {
				int64_t i_new_source_vertex = new_vertex_index_data[i_source_vertex];
				auto vertex_edges = edge_data + i_source_vertex * max_vertex_degree;
				auto new_vertex_edges = new_edge_data + i_new_source_vertex * new_max_vertex_degree;
				for (int i_edge = 0; i_edge < max_vertex_degree; i_edge++) {
					int64_t i_target_vertex = vertex_edges[i_edge];
					int64_t i_new_target_vertex = new_vertex_index_data[i_target_vertex];
					bool already_added = false;
					for (int i_new_edge = 0; i_new_edge < new_edge_counts.GetCount(i_new_source_vertex); i_new_edge++) {
						if (new_vertex_edges[i_new_edge] == i_new_target_vertex) {
							already_added = true;
							break;
						}
					}
					//TODO: take care of possible race condition -- adding two equivalent i_new_target_vertex values from separate threads
					if (!already_added) {
						int new_edge_index = new_edge_counts.FetchAdd(i_new_source_vertex, 1);
						if (new_edge_index < new_max_vertex_degree) {
							new_vertex_edges[new_edge_index] = i_new_target_vertex;
						}
					}
				}
			}
	);

}


} // namespace nnrt::geometry::functional::kernel::sampling