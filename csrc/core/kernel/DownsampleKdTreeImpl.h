//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/3/22.
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

#include "core/kernel/KdTree.h"
#include "core/kernel/KdTreeUtilities.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentAtomics.h"

//__DEBUG
// #define __CUDACC__

#ifdef __CUDACC__
#include <stdgpu/atomic.cuh>
#else

#include <atomic>

#endif


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;


namespace nnrt::core::kernel::kdtree {
namespace {

#define NEIGHBOR_SPACE_CHECK

template<o3c::Device::DeviceType TDeviceType, typename TPoint, typename TMakePoint>
NNRT_DEVICE_WHEN_CUDACC
inline void FindRadiusNeighbors_KdTree(int32_t* radius_neighbor_indices, float* radius_neighbor_distances, const KdTreeNode* nodes,
                                      const int node_count, float radius, const TPoint& query_point, int32_t query_point_index,
                                      const NDArrayIndexer& reference_point_indexer, TMakePoint&& make_point) {


	// Allocate traversal stack from thread-local memory,
	// and push nullptr onto the stack to indicate that there are no deferred nodes.
	int32_t stack[NNRT_KDTREE_STACK_SIZE];
	int32_t* stack_cursor = stack;
	*stack_cursor = -1; // push "-1" onto the bottom of the stack
	stack_cursor++; // advance the stack cursor

	int32_t node_index = 0;
	int32_t neighbor_count = 0;
	do {
		const kdtree::KdTreeNode& node = nodes[node_index];
		if (node_index < node_count && !node.Empty() && node.point_index != query_point_index) {
			auto node_point = make_point(reference_point_indexer.template GetDataPtr<float>(node.point_index));

			float node_distance = (node_point - query_point).norm();
			// update the nearest neighbor heap if the distance is better than the greatest nearest-neighbor distance encountered so far
			if (radius >= node_distance) {
				radius_neighbor_indices[neighbor_count] = node.point_index;
				radius_neighbor_distances[neighbor_count] = node_distance;
				neighbor_count++; //just cross fingers and hope we don't run out of space here
#ifdef NEIGHBOR_SPACE_CHECK
				if (neighbor_count == NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS) {
					neighbor_count--;
					printf("WARNING: exceeded maximum number of radius neighbors. Please pick a smaller radius for this point density.\n");
				}
#endif
			}

			const uint8_t i_dimension = node.i_split_dimension;
			float split_plane_value = node_point.coeff(i_dimension);
			float query_coordinate = query_point[i_dimension];

			// Query overlaps an internal node => traverse.
			const int32_t left_child_index = kdtree::GetLeftChildIndex(node_index);
			const int32_t right_child_index = kdtree::GetRightChildIndex(node_index);

			bool search_left_first = query_coordinate < split_plane_value;
			bool search_left = false;
			if (query_coordinate - radius <= split_plane_value) {
				// circle with max_knn_distance radius around the query point overlaps the left subtree
				search_left = true;
			}
			bool search_right = false;
			if (query_coordinate + radius > split_plane_value) {
				// circle with max_knn_distance radius around the query point overlaps the right subtree
				search_right = true;
			}

			if (!search_left && !search_right) {
				// pop from stack: (1) move cursor back to point at previous entry in the stack, (2) dereference
				node_index = *(--stack_cursor);
			} else {
				if (search_left_first) {
					node_index = search_left ? left_child_index : right_child_index;
					if (search_left && search_right) {
						// push right child onto the stack at the current cursor position
						*stack_cursor = right_child_index;
						stack_cursor++; // advance the stack cursor
					}
				} else {
					node_index = search_right ? right_child_index : left_child_index;
					if (search_left && search_right) {
						// push left child onto the stack at the current cursor position
						*stack_cursor = left_child_index;
						stack_cursor++; // advance the stack cursor
					}
				}
			}
		} else {
			node_index = *(--stack_cursor);
		}
	} while (node_index != -1);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TMakePoint>
void DecimateReferencePoints_Generic(
		open3d::core::Tensor& decimated_points, const KdTreeNode* nodes, int node_count,
		const open3d::core::Tensor& reference_points, float downsampling_radius, TMakePoint&& make_point) {
	const int64_t reference_point_count = reference_points.GetShape(0);

	auto radius_neighbors = o3c::Tensor({reference_point_count, NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS}, o3c::Dtype::Int32,
	                                    reference_points.GetDevice());
	auto radius_neighbor_distances = o3c::Tensor({reference_point_count, NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS}, o3c::Dtype::Float32,
	                                             reference_points.GetDevice());
	radius_neighbors.template Fill(-1);
	NDArrayIndexer radius_neighbor_indexer(radius_neighbors, 1);
	NDArrayIndexer radius_neighbor_distance_indexer(radius_neighbor_distances, 1);
	NDArrayIndexer reference_point_indexer(reference_points, 1);

	// find fixed-radius neighbors for each point
	open3d::core::ParallelFor(
			reference_points.GetDevice(), reference_point_count,
			[=]
					OPEN3D_DEVICE(int64_t
					              workload_idx) {
				auto* point_radius_neighbors = radius_neighbor_indexer.GetDataPtr<int>(workload_idx);
				auto* point_radius_neighbor_distances = radius_neighbor_distance_indexer.GetDataPtr<float>(workload_idx);
				auto query_point = make_point(reference_point_indexer.template GetDataPtr<float>(workload_idx));
				FindRadiusNeighbors_KdTree<TDeviceType>(point_radius_neighbors, point_radius_neighbor_distances, nodes, node_count,
				                                        downsampling_radius, query_point, (int32_t) workload_idx, reference_point_indexer,
														make_point);
			}
	);


	auto point_mask = o3c::Tensor({reference_point_count}, o3c::Dtype::Bool, reference_points.GetDevice());
	point_mask.template Fill(true);
	bool* point_mask_data = reinterpret_cast<bool*> (point_mask.GetDataPtr());

	auto filtered_points = open3d::core::Tensor({reference_point_count}, o3c::Dtype::Int32, reference_points.GetDevice());
	int* filtered_point_data = reinterpret_cast<int*>(filtered_points.GetDataPtr());

	DECLARE_ATOMIC(int, candidate_filtered_point_count);
	INITIALIZE_ATOMIC(int, candidate_filtered_point_count, 0);
	DECLARE_ATOMIC(int, ready_filtered_point_count);
	INITIALIZE_ATOMIC(int, candidate_filtered_point_count, 0);

	float downsampling_radius_squared = downsampling_radius * downsampling_radius;

	// find the subset of points such that equal circles drawn with each point as the center are not overlapping
	open3d::core::ParallelFor(
			reference_points.GetDevice(), reference_point_count,
#ifdef __CUDACC__
			[=]
#else
			[&]
#endif
					OPEN3D_DEVICE(int64_t
					              workload_idx) {
				auto query_point = make_point(reference_point_indexer.template GetDataPtr<float>(workload_idx));
				int filtered_point_count = 0;
				int i_filtered_point = 0;
				do {
					do {
						// wait until the points that are currently being added to filtered point array finish being added
						filtered_point_count = GET_ATOMIC_VALUE(candidate_filtered_point_count);
					} while (filtered_point_count < GET_ATOMIC_VALUE(ready_filtered_point_count) && point_mask_data[workload_idx]);
					for (; i_filtered_point < filtered_point_count; i_filtered_point++) {
						if (!point_mask_data[workload_idx]) {
							return;
						}
						auto filtered_point = make_point(reference_point_indexer.template GetDataPtr<float>(filtered_point_data[i_filtered_point]));
						// if the point is within the radius of another point in the filtered list, we can safely ignore.
						if ((filtered_point - query_point).squaredNorm() < downsampling_radius_squared) {
							return;
						}
					}
				} while (ATOMIC_CE(candidate_filtered_point_count, filtered_point_count, filtered_point_count + 1));
				// we now know for certain this point's radius doesn't overlap with other filtered points' radii
				filtered_point_data[filtered_point_count] = workload_idx;
				// atomically increment the counter to indicate that the point's index has been added to the filtered point ledger
				ATOMIC_ADD(ready_filtered_point_count, 1);

				auto* point_radius_neighbors = radius_neighbor_indexer.GetDataPtr<int>(workload_idx);
				int i_neighbor = 0;
				while (i_neighbor < NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS && point_radius_neighbors[i_neighbor] != -1) {
					point_mask_data[point_radius_neighbors[i_neighbor]] = false;
				}

			}
	);
	decimated_points = reference_points.GetItem(o3c::TensorKey::IndexTensor(point_mask));
}

} // anonymous namespace

template<open3d::core::Device::DeviceType TDeviceType>
void DecimateReferencePoints(open3d::core::Tensor& decimated_points, open3d::core::Blob& index_data, int node_count,
                             const open3d::core::Tensor& reference_points, float downsampling_radius) {
	auto dimension_count = (int32_t) reference_points.GetShape(1);
	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	switch (dimension_count) {
		case 1:
			DecimateReferencePoints_Generic<TDeviceType>(
					decimated_points, nodes, node_count, reference_points, downsampling_radius,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, 1>>(vector_data, dimension_count);
					}
			);
			break;
		case 2:
			DecimateReferencePoints_Generic<TDeviceType>(
					decimated_points, nodes, node_count, reference_points, downsampling_radius,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector2f>(vector_data, dimension_count, 1);
					}
			);
			break;
		case 3:
			DecimateReferencePoints_Generic<TDeviceType>(
					decimated_points, nodes, node_count, reference_points, downsampling_radius,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector3f>(vector_data, dimension_count, 1);
					}
			);
			break;
		default:
			DecimateReferencePoints_Generic<TDeviceType>(
					decimated_points, nodes, node_count, reference_points, downsampling_radius,
					[dimension_count] NNRT_DEVICE_WHEN_CUDACC(float* vector_data) {
						return Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(vector_data, dimension_count);
					}
			);

	}
}

} // namespace nnrt::core::kernel::kdtree