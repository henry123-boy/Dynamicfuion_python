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
#include <open3d/core/kernel/CUDALauncher.cuh>

#include "core/CUDA/DeviceHeapCUDA.cuh"
#include "core/kernel/BuildKdTreeImpl.h"
#include "core/kernel/SearchKdTreeImpl.h"

namespace nnrt::core::kernel::kdtree {

template
void BuildKdTreeIndex<open3d::core::Device::DeviceType::CUDA>(open3d::core::Blob& index_data, const open3d::core::Tensor& points, void** root);

template
void FindKNearestKdTreePoints<open3d::core::Device::DeviceType::CUDA>(open3d::core::Tensor& closest_indices, open3d::core::Tensor& squared_distances,
                                                                      const open3d::core::Tensor& query_points,
                                                                      int32_t k, const open3d::core::Blob& index_data,
                                                                      const open3d::core::Tensor& kd_tree_points, const void* root);

void IndexDataToHost_CUDA(open3d::core::Blob& index_data_cpu, const open3d::core::Blob& index_data, int point_count) {
	auto* nodes_cpu = reinterpret_cast<KdTreeNode*>(index_data_cpu.GetDataPtr());
	auto* nodes = reinterpret_cast<const KdTreeNode*>(index_data.GetDataPtr());

	namespace launcher = o3c::kernel::cuda_launcher;

	o3c::Tensor child_indices({point_count, 2}, o3c::Dtype::Int32, index_data.GetDevice());
	o3gk::NDArrayIndexer child_index_indexer(child_indices, 1);

	launcher::ParallelFor(
			point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				const KdTreeNode& node = nodes[workload_idx];
				auto* child_set = child_index_indexer.GetDataPtr<int32_t>(workload_idx);
				if (node.left_child == nullptr) {
					child_set[0] = -1;
				} else {
					child_set[0] = static_cast<int32_t>(node.left_child - nodes);
				}
				if (node.right_child == nullptr) {
					child_set[1] = -1;
				} else {
					child_set[1] = static_cast<int32_t>(node.right_child - nodes);
				}
			}
	);
	auto host_child_indices = child_indices.To(o3c::Device("CPU:0"));
	o3gk::NDArrayIndexer host_child_index_indexer(host_child_indices, 1);
	for (int i_node = 0; i_node < point_count; i_node++) {
		auto children_offsets = host_child_index_indexer.GetDataPtr<int32_t>(i_node);
		if (children_offsets[0] == -1) {
			nodes_cpu[i_node].left_child = nullptr;
		} else {
			nodes_cpu[i_node].left_child = nodes_cpu + children_offsets[0];
		}
		if (children_offsets[1] == -1) {
			nodes_cpu[i_node].right_child = nullptr;
		} else {
			nodes_cpu[i_node].right_child = nodes_cpu + children_offsets[1];
		}
	}
}

} // nnrt::core::kernel