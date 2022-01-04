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
#include "core/kernel/KdTreeUtils.h"
#include "core/DeviceSelection.h"
#include "core/DimensionCount.h"


namespace o3gk = open3d::t::geometry::kernel;
namespace o3c = open3d::core;

namespace nnrt::core::kernel::kdtree {

void BuildKdTreeIndex(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root, int& root_node_index) {
	core::InferDeviceFromEntityAndExecute(
			points,
			[&] { BuildKdTreeIndex<o3c::Device::DeviceType::CPU>(index_data, points, root, root_node_index); },
			[&] { NNRT_IF_CUDA(BuildKdTreeIndex<o3c::Device::DeviceType::CUDA>(index_data, points, root, root_node_index);); }
	);

}

void
FindKNearestKdTreePoints(open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
                         const open3d::core::Tensor& query_points,
                         int32_t k, const open3d::core::Blob& index_data, const open3d::core::Tensor& kd_tree_points) {
	const int dimension_count = (int) kd_tree_points.GetShape(1);
	core::InferDeviceFromEntityAndExecute(
			kd_tree_points,
			[&] {
				FindKNearestKdTreePoints<o3c::Device::DeviceType::CPU>(
						closest_indices, squared_distances, query_points, k, index_data, kd_tree_points);
			},
			[&] {
				NNRT_IF_CUDA(
						FindKNearestKdTreePoints<o3c::Device::DeviceType::CUDA>(
								closest_indices, squared_distances, query_points, k, index_data, kd_tree_points);
				);
			}
	);
}

inline open3d::core::Blob BlobToDevice(const open3d::core::Blob& index_data, int64_t byte_count, const o3c::Device& device) {
	o3c::Blob target_blob(byte_count, device);
	o3c::MemoryManager::Memcpy(target_blob.GetDataPtr(), device, index_data.GetDataPtr(), index_data.GetDevice(), byte_count);
	return target_blob;
}

void GenerateTreeDiagram(std::string& diagram, const open3d::core::Blob& index_data, const void* root, const open3d::core::Tensor& kd_tree_points) {
	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());
	const auto* root_node = reinterpret_cast<const KdTreeNode*>(root);
	auto root_index = static_cast<int32_t>(root_node - nodes);
	auto node_count = kd_tree_points.GetLength();
	auto index_data_cpu = BlobToDevice(index_data, node_count * static_cast<int64_t>(sizeof(kernel::kdtree::KdTreeNode)), o3c::Device("CPU:0"));
	auto* nodes_cpu = reinterpret_cast<KdTreeNode*>(index_data_cpu.GetDataPtr());
	KdTreeNode* root_node_cpu = nodes_cpu + root_index;
	std::function<int32_t(KdTreeNode*, int32_t)> get_tree_height = [&get_tree_height](KdTreeNode* node, int32_t height) {
		if (node != nullptr) {
			int32_t left_subtree_height = 1 + get_tree_height(node->left_child, height);
			int32_t right_subtree_height = 1 + get_tree_height(node->right_child, height);
			return std::max(left_subtree_height, right_subtree_height);
		}
		return 0;
	};
	int32_t tree_height = get_tree_height(root_node_cpu, 0);
	const int dimension_count = static_cast<int>(kd_tree_points.GetShape(1));
	const int digit_length = 5;
	const int coordinate_spacing = 1;


	o3gk::NDArrayIndexer point_indexer(kd_tree_points, 1);

	std::function<std::string(int)> point_to_string = [&point_indexer, &dimension_count](int point_index) {
		auto* point_data = point_indexer.GetDataPtr<float>(point_index);
		std::string point_string = fmt::format("[{:0>5}", point_data[0]);
		for (int i_dimension = 1; i_dimension < dimension_count; i_dimension++) {
			point_string += fmt::format(" {:0>5}", point_data[i_dimension]);
		}
		point_string += fmt::format(" {:0>5}]", point_data[dimension_count - 1]);
		return point_string;
	};
	const int point_string_length = (digit_length + coordinate_spacing) * (dimension_count - 1) + digit_length + 2; //2 is for the brackets
	const int min_gap_size = 1;

	const int leaf_node_length = point_string_length + min_gap_size;
	int last_level_string_length = IntPower(2, tree_height - 1) * leaf_node_length + 1;
	auto get_initial_offset = [&last_level_string_length, &leaf_node_length](int level) {
		return (last_level_string_length / (2 * IntPower(2, level))) - (leaf_node_length / 2);
	};

	auto get_gap_length = [&last_level_string_length, &leaf_node_length](int level) {
		return (last_level_string_length / IntPower(2, level)) - leaf_node_length + 1;
	};

	const int row_count = tree_height * 3 - 2;
	std::vector<std::string> row_strings(row_count);
	bool level_initialized[tree_height];

	std::function<void(KdTreeNode*, int)> fill_tree_level_strings =
			[&fill_tree_level_strings, &get_initial_offset, &get_gap_length, &row_strings, &level_initialized, &point_to_string]
					(KdTreeNode* node, int level) {
				fill_tree_level_strings(node->left_child, level + 1);
				fill_tree_level_strings(node->right_child, level + 1);
				const int point_row = level * 3;
				if (level_initialized[level]) {
					row_strings[point_row] += fmt::format("{: >{}}", "", get_gap_length(level));
				} else {
					int initial_offset = get_initial_offset(level);
					row_strings[point_row] += fmt::format("{: >{}}", "", initial_offset);
					level_initialized[level] = true;
				}
				row_strings[level * 3] += point_to_string(node->index);
			};
	fill_tree_level_strings(root_node_cpu, 0);
	diagram = string::join(row_strings, "\n");
}

} //  nnrt::core::kernel::kdtree
