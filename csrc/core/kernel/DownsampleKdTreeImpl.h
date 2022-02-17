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


namespace o3c = open3d::core;
namespace o3gk = open3d::t::geometry::kernel;


namespace nnrt::core::kernel::kdtree {
namespace {

#define NEIGHBOR_SPACE_CHECK

template<o3c::Device::DeviceType TDeviceType, typename TPoint>
NNRT_DEVICE_WHEN_CUDACC
inline int FindRadiusNeighbors_KdTree(int32_t* radius_neighbor_indices, float* radius_neighbor_distances, const KdTreeNode* nodes,
                                      const int node_count, float radius, const TPoint& query_point, int32_t query_point_index,
                                      const NDArrayIndexer& reference_point_indexer) {


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
			Eigen::Map<Eigen::Vector3f> node_point(reference_point_indexer.template GetDataPtr<float>(node.point_index));

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

	// auto radius_neighbor_counts = o3c::Tensor({reference_point_count}, o3c::Dtype::Int32, reference_points.GetDevice());
	auto radius_neighbors = o3c::Tensor({reference_point_count, NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS}, o3c::Dtype::Int32,
	                                    reference_points.GetDevice());
	auto radius_neighbor_distances = o3c::Tensor({reference_point_count, NNRT_KDTREE_MAX_EXPECTED_RADIUS_NEIGHBORS}, o3c::Dtype::Float32,
	                                    reference_points.GetDevice());
	radius_neighbors.template Fill(-1);
	NDArrayIndexer radius_neighbor_indexer(radius_neighbors, 1);
	NDArrayIndexer reference_point_indexer(reference_points, 1);

	open3d::core::ParallelFor(
			reference_points.GetDevice(), reference_point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto* point_radius_neighbors = radius_neighbor_indexer.GetDataPtr<int>(workload_idx);
				auto query_point = make_point(reference_point_indexer.template GetDataPtr<float>(workload_idx));
				FindRadiusNeighbors_KdTree(point_radius_neighbors, radius_neighbor_distances, nodes, node_count, downsampling_radius, query_point, (int32_t) workload_idx,
				                           reference_point_indexer);
			}
	);


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