//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/25/21.
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
#include <fmt/format.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

#include "string/join_string_separator.h"
#include "core/kernel/KdTree.h"
#include "core/kernel/KdTreeUtilities.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "core/DeviceSelection.h"
#include "core/DimensionCount.h"


namespace o3gk = open3d::t::geometry::kernel;
namespace o3c = open3d::core;

namespace nnrt::core::kernel::kdtree {

void BuildKdTreeIndex(open3d::core::Blob& index_data, int64_t index_length, const open3d::core::Tensor& points) {
	core::InferDeviceFromEntityAndExecute(
			points,
			[&] { BuildKdTreeIndex<o3c::Device::DeviceType::CPU>(index_data, index_length, points); },
			[&] { NNRT_IF_CUDA(BuildKdTreeIndex<o3c::Device::DeviceType::CUDA>(index_data, index_length, points);); }
	);

}

template<NeighborTrackingStrategy TTrackingStrategy>
void FindKNearestKdTreePoints(open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
                              open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
                              const open3d::core::Tensor& reference_points) {
	core::InferDeviceFromEntityAndExecute(
			reference_points,
			[&] {
				FindKNearestKdTreePoints<o3c::Device::DeviceType::CPU, TTrackingStrategy>(
						index_data, index_length, nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, reference_points
				);
			},
			[&] {
				NNRT_IF_CUDA(
						FindKNearestKdTreePoints<o3c::Device::DeviceType::CUDA, TTrackingStrategy>(
								index_data, index_length, nearest_neighbor_indices, nearest_neighbor_distances, query_points, k, reference_points
						);
				);
			}
	);
}

template void FindKNearestKdTreePoints<NeighborTrackingStrategy::PLAIN>(
		open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
		open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
		const open3d::core::Tensor& reference_points
);

template
void FindKNearestKdTreePoints<NeighborTrackingStrategy::PRIORITY_QUEUE>(
		open3d::core::Blob& index_data, int index_length, open3d::core::Tensor& nearest_neighbor_indices,
		open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points, int32_t k,
		const open3d::core::Tensor& reference_points
);


void GenerateTreeDiagram(std::string& diagram, const open3d::core::Blob& index_data, const int index_length,
                         const open3d::core::Tensor& kd_tree_points, const int digit_length) {

	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	auto node_count = kd_tree_points.GetLength();
	auto index_data_cpu = BlobToDevice(index_data, index_length * static_cast<int64_t>(sizeof(KdTreeNode)), o3c::Device("CPU:0"));
	auto* nodes_cpu = reinterpret_cast<KdTreeNode*>(index_data_cpu.GetDataPtr());

	std::function<int32_t(int32_t, int32_t)> get_tree_height =
			[&get_tree_height, &nodes_cpu, &index_length](int32_t node_index, int32_t height) {
				if (node_index < index_length) {
					KdTreeNode& node = nodes_cpu[node_index];
					if (!node.Empty()) {
						int32_t left_subtree_height = 1 + get_tree_height(GetLeftChildIndex(node_index), height);
						int32_t right_subtree_height = 1 + get_tree_height(GetRightChildIndex(node_index), height);
						return std::max(left_subtree_height, right_subtree_height);
					}
				}
				return 0;
			};
	int32_t tree_height = get_tree_height(0, 0);
	const int dimension_count = static_cast<int>(kd_tree_points.GetShape(1));
	const int coordinate_spacing = 1;

	o3c::Tensor kd_tree_points_cpu = kd_tree_points.To(o3c::Device("CPU:0"), false);


	o3gk::NDArrayIndexer point_indexer(kd_tree_points_cpu, 1);

	std::function<std::string(int)> point_to_string = [&point_indexer, &dimension_count, & digit_length](int point_index) {
		auto* point_data = point_indexer.GetDataPtr<float>(point_index);
		std::string point_string = fmt::format("[{: >{}}", point_data[0], digit_length);
		for (int i_dimension = 1; i_dimension < dimension_count; i_dimension++) {
			point_string += fmt::format(" {: >{}}", point_data[i_dimension], digit_length);
		}
		point_string += "]";
		return point_string;
	};
	const int point_string_length =
			(digit_length + coordinate_spacing) * (dimension_count - 1) + digit_length + 2; //2 is for the brackets, 1 for the sign
	const int min_gap_size = 1;

	const int leaf_node_length = point_string_length + min_gap_size;
	int last_level_string_length = IntPower(2, tree_height - 1) * leaf_node_length;
	auto get_initial_offset = [&leaf_node_length, &tree_height](int level) {
		return (IntPower(2, tree_height - level - 1) - 1) * (leaf_node_length / 2);
	};

	auto get_gap_length = [&leaf_node_length, &tree_height](int level) {
		return (IntPower(2, tree_height - level - 1) - 1) * leaf_node_length + 1;
	};

	const int row_count = tree_height * 3 - 2;
	std::vector<std::string> row_strings(row_count);
	// initialize all rows with dimension
	int i_dimension = 0;
	const int dim_digit_length = 1;
	const std::string dim_prefix = "dim: ";
	for (int i_row = 0; i_row < row_count; i_row++) {
		if (i_row % 3 == 0) {
			// we want some descriptor of the kind "dim: <dimension>" at each point row
			row_strings[i_row] += fmt::format("{}{: >{}} ", dim_prefix, i_dimension, dim_digit_length);
			i_dimension = (i_dimension + 1) % dimension_count;
		} else {
			// initialize non-point rows with white space of length equivalent to dimension prefix & digit
			row_strings[i_row] += fmt::format("{: >{}} ", "", dim_prefix.length() + dim_digit_length);
		}
	}
	bool level_initialized[tree_height];
	std::fill(level_initialized, level_initialized + tree_height, 0);


	std::function<void(int, int, bool)> fill_tree_level_strings =
			[&fill_tree_level_strings, &get_initial_offset, &get_gap_length, &row_strings,
					&level_initialized, &point_to_string, &point_string_length, &tree_height, &nodes_cpu,
					&index_length]
					(int node_index, int level, bool right) {
				if (node_index >= index_length) return;
				KdTreeNode& node = nodes_cpu[node_index];
				if (!node.Empty()) {
					fill_tree_level_strings(GetLeftChildIndex(node_index), level + 1, false);
					fill_tree_level_strings(GetRightChildIndex(node_index), level + 1, true);
					const int point_row = level * 3;
					if (level_initialized[level]) {
						const int gap_length = get_gap_length(level);
						row_strings[point_row] += fmt::format("{: >{}}", "", gap_length);
						if (level > 0) {
							if (right) {
								const int diagonal_bar_at = gap_length + 2 * (point_string_length / 2 - 1);
								const int diagonal_bar_row = level * 3 - 1;
								row_strings[diagonal_bar_row] += fmt::format("{: >{}}\\", "", diagonal_bar_at);
								const int handle_row = level * 3 - 2;
								const int vertical_bar_at = diagonal_bar_at / 2;
								row_strings[handle_row] += fmt::format("{:_>{}}", "", vertical_bar_at);
							} else {
								const int diagonal_bar_at = gap_length + 2 * (point_string_length / 2 + 1);
								const int diagonal_bar_row = level * 3 - 1;
								row_strings[diagonal_bar_row] += fmt::format("{: >{}}/", "", diagonal_bar_at);
								const int handle_row = level * 3 - 2;
								row_strings[handle_row] += fmt::format("{: >{}}", "", diagonal_bar_at + 2);
								const int vertical_bar_at = gap_length / 2 + (point_string_length / 2 - 1);
								row_strings[handle_row] += fmt::format("{:_>{}}|", "", vertical_bar_at);
							}
						}
					} else {
						const int initial_offset = get_initial_offset(level);
						row_strings[point_row] += fmt::format("{: >{}}", "", initial_offset);
						level_initialized[level] = true;
						// draw the connector to parent (can only be a left child, since whole level just initialized)
						if (level > 0) {
							const int parent_level_initial_offset = get_initial_offset(level - 1);
							const int diagonal_bar_at = initial_offset + (point_string_length / 2 + 1);
							const int diagonal_bar_row = level * 3 - 1;
							row_strings[diagonal_bar_row] += fmt::format("{: >{}}/", "", diagonal_bar_at);
							const int handle_row = level * 3 - 2;
							row_strings[handle_row] += fmt::format("{: >{}}", "", diagonal_bar_at + 1);
							const int vertical_bar_at = parent_level_initial_offset - initial_offset - 2;
							row_strings[handle_row] += fmt::format("{:_>{}}|", "", vertical_bar_at);
						}
					}
					row_strings[point_row] += point_to_string(node.point_index);
				} else if (level < tree_height) {
					// we have to print empty space here when nodes are missing from some tree branches
					const int point_row = level * 3;
					if (level_initialized[level]) {
						const int gap_length = get_gap_length(level);
						row_strings[point_row] += fmt::format("{: >{}}", "", gap_length);
						if (level > 0) {
							if (right) {
								const int diagonal_bar_at = gap_length + 2 * (point_string_length / 2 - 1);
								const int diagonal_bar_row = level * 3 - 1;
								row_strings[diagonal_bar_row] += fmt::format("{: >{}} ", "", diagonal_bar_at);
								const int handle_row = level * 3 - 2;
								const int vertical_bar_at = diagonal_bar_at / 2;
								row_strings[handle_row] += fmt::format("{: >{}}", "", vertical_bar_at);
							} else {
								const int diagonal_bar_at = gap_length + 2 * (point_string_length / 2 + 1);
								const int diagonal_bar_row = level * 3 - 1;
								row_strings[diagonal_bar_row] += fmt::format("{: >{}} ", "", diagonal_bar_at);
								const int handle_row = level * 3 - 2;
								row_strings[handle_row] += fmt::format("{: >{}}", "", diagonal_bar_at + 2);
								const int vertical_bar_at = gap_length / 2 + (point_string_length / 2 - 1);
								row_strings[handle_row] += fmt::format("{: >{}} ", "", vertical_bar_at);
							}
						}
					} else {
						const int initial_offset = get_initial_offset(level);
						row_strings[point_row] += fmt::format("{: >{}}", "", initial_offset);
						level_initialized[level] = true;
						// draw the connector to parent (can only be a left child, since whole level just initialized)
						if (level > 0) {
							const int parent_level_initial_offset = get_initial_offset(level - 1);
							const int diagonal_bar_at = initial_offset + (point_string_length / 2 + 1);
							const int diagonal_bar_row = level * 3 - 1;
							row_strings[diagonal_bar_row] += fmt::format("{: >{}} ", "", diagonal_bar_at);
							const int handle_row = level * 3 - 2;
							row_strings[handle_row] += fmt::format("{: >{}}", "", diagonal_bar_at + 1);
							const int vertical_bar_at = parent_level_initial_offset - initial_offset - 2;
							row_strings[handle_row] += fmt::format("{: >{}} ", "", vertical_bar_at);
						}
					}
					row_strings[point_row] += fmt::format("{: >{}}", "", point_string_length);
				}
			};
	fill_tree_level_strings(0, 0, false);
	diagram = string::join(row_strings, "\n");
}


} //  nnrt::core::kernel::kdtree
