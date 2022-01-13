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
#include "core/kernel/KdTreePointCloud.h"
#include "core/kernel/KdTreeUtils.h"
#include "core/DeviceSelection.h"
#include "core/DimensionCount.h"


namespace o3gk = open3d::t::geometry::kernel;
namespace o3c = open3d::core;

namespace nnrt::core::kernel::kdtree {

size_t GetNodeByteCount(const open3d::core::Tensor& points) {
	switch (points.GetShape(1)) {
		case 1:
			return sizeof(KdTreePointCloudNode<Eigen::Vector<float, 1>>);
		case 2:
			return sizeof(KdTreePointCloudNode<Eigen::Vector2f>);
		case 3:
			return sizeof(KdTreePointCloudNode<Eigen::Vector3f>);
		default:
			return sizeof(KdTreePointCloudNode<Eigen::Vector<float, Eigen::Dynamic>>);
	}
}

void BuildKdTreePointCloud(open3d::core::Blob& node_data, const open3d::core::Tensor& points, void** root) {
	core::InferDeviceFromEntityAndExecute(
			points,
			[&] { BuildKdTreePointCloud<o3c::Device::DeviceType::CPU>(node_data, points, root); },
			[&] { NNRT_IF_CUDA(BuildKdTreePointCloud<o3c::Device::DeviceType::CUDA>(node_data, points, root);); }
	);

}

template<SearchStrategy TSearchStrategy>
void FindKNearestKdTreePointCloudPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances, const open3d::core::Tensor& query_points,
                                        const int32_t k, const void* root, int dimension_count) {
	core::InferDeviceFromEntityAndExecute(
			query_points,
			[&] {
				FindKNearestKdTreePointCloudPoints<o3c::Device::DeviceType::CPU, TSearchStrategy>(
						nearest_neighbors, nearest_neighbor_distances, query_points, k, root, dimension_count);
			},
			[&] {
				NNRT_IF_CUDA(
						FindKNearestKdTreePointCloudPoints<o3c::Device::DeviceType::CUDA, TSearchStrategy>(
								nearest_neighbors, nearest_neighbor_distances, query_points, k, root, dimension_count);
				);
			}
	);
}

template
void FindKNearestKdTreePointCloudPoints<SearchStrategy::ITERATIVE>(open3d::core::Tensor& nearest_neighbors,
                                                         open3d::core::Tensor& nearest_neighbor_distances,
                                                         const open3d::core::Tensor& query_points, int32_t k,
                                                         const void* root, int dimension_count);


template
void FindKNearestKdTreePointCloudPoints<SearchStrategy::RECURSIVE>(open3d::core::Tensor& nearest_neighbors,
                                                         open3d::core::Tensor& nearest_neighbor_distances,
                                                         const open3d::core::Tensor& query_points, int32_t k,
                                                         const void* root, int dimension_count);


template<typename TPoint>
void GenerateKdTreePointCloudDiagram(std::string& diagram, const open3d::core::Blob& node_data, const void* root,
                                     const int point_count, const int dimension_count, const int digit_length) {

	auto* nodes = reinterpret_cast<const KdTreePointCloudNode<TPoint>*>(node_data.GetDataPtr());
	const auto* root_node = reinterpret_cast<const KdTreePointCloudNode<TPoint>*>(root);
	auto root_index = static_cast<int32_t>(root_node - nodes);
	auto node_data_cpu = PointCloudDataToHost<TPoint>(node_data, point_count, dimension_count);
	auto* nodes_cpu = reinterpret_cast<KdTreePointCloudNode<TPoint>*>(node_data_cpu.GetDataPtr());
	KdTreePointCloudNode<TPoint>* root_node_cpu = nodes_cpu + root_index;
	std::function<int32_t(KdTreePointCloudNode<TPoint>*, int32_t)> get_tree_height = [&get_tree_height](KdTreePointCloudNode<TPoint>* node,
	                                                                                                    int32_t height) {
		if (node != nullptr) {
			int32_t left_subtree_height = 1 + get_tree_height(node->left_child, height);
			int32_t right_subtree_height = 1 + get_tree_height(node->right_child, height);
			return std::max(left_subtree_height, right_subtree_height);
		}
		return 0;
	};
	int32_t tree_height = get_tree_height(root_node_cpu, 0);
	const int coordinate_spacing = 1;

	std::function<std::string(const TPoint&)> point_to_string = [&dimension_count, & digit_length](const TPoint& point) {
		std::string point_string = fmt::format("[{: >{}}", point.coeff(0), digit_length);
		for (int i_dimension = 1; i_dimension < dimension_count; i_dimension++) {
			point_string += fmt::format(" {: >{}}", point.coeff(i_dimension), digit_length);
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


	std::function<void(KdTreePointCloudNode<TPoint>*, int, bool)> fill_tree_level_strings =
			[&fill_tree_level_strings, &get_initial_offset, &get_gap_length, &row_strings,
					&level_initialized, &point_to_string, &point_string_length, &tree_height]
					(KdTreePointCloudNode<TPoint>* node, int level, bool right) {
				if (node != nullptr) {
					fill_tree_level_strings(node->left_child, level + 1, false);
					fill_tree_level_strings(node->right_child, level + 1, true);
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
					row_strings[point_row] += point_to_string(node->point);
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
	fill_tree_level_strings(root_node_cpu, 0, false);
	diagram = string::join(row_strings, "\n");
}

void GenerateKdTreePointCloudDiagram(std::string& diagram, const open3d::core::Blob& kd_tree_point_cloud_data, const void* root, int point_count,
                                     int dimension_count, int digit_length) {
	switch (dimension_count) {
		case 1:
			GenerateKdTreePointCloudDiagram<Eigen::Vector<float, 1>>(diagram, kd_tree_point_cloud_data,
			                                                         root, point_count, dimension_count, digit_length);
			break;
		case 2:
			GenerateKdTreePointCloudDiagram<Eigen::Vector2f>(diagram, kd_tree_point_cloud_data,
			                                                 root, point_count, dimension_count, digit_length);
			break;
		case 3:
			GenerateKdTreePointCloudDiagram<Eigen::Vector3f>(diagram, kd_tree_point_cloud_data,
			                                                 root, point_count, dimension_count, digit_length);
			break;
		default:
			GenerateKdTreePointCloudDiagram<Eigen::Vector<float, Eigen::Dynamic>>(diagram, kd_tree_point_cloud_data,
			                                                                      root, point_count, dimension_count, digit_length);
			break;
	}

}

template<typename TPoint>
open3d::core::Blob PointCloudDataToHost(const open3d::core::Blob& index_data, int point_count, int dimension_count) {
	o3c::Blob index_data_cpu = BlobToDevice(index_data, point_count * static_cast<int64_t>(sizeof(KdTreePointCloudNode<TPoint>)),
	                                        o3c::Device("CPU:0"));
	core::InferDeviceFromEntityAndExecute(
			index_data,
			[&] {},
			[&] { NNRT_IF_CUDA(PointCloudDataToHost_CUDA(index_data_cpu, index_data, point_count, dimension_count);); }
	);
	return index_data_cpu;
}


} //  nnrt::core::kernel::kdtree
