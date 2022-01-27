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
#include <open3d/core/ParallelFor.h>

#include "core/CUDA/DeviceHeapCUDA.cuh"
#include "core/kernel/BuildKdTreePointCloudImpl.h"
#include "core/kernel/SearchKdTreePointCloudImpl.h"

namespace nnrt::core::kernel::kdtree {

template
void BuildKdTreePointCloud<open3d::core::Device::DeviceType::CUDA>(open3d::core::Blob& node_data, const open3d::core::Tensor& points, void** root);

template
void FindKNearestKdTreePointCloudPoints<open3d::core::Device::DeviceType::CUDA, SearchStrategy::ITERATIVE>(
		open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
		const open3d::core::Tensor& query_points, int32_t k, const void* root, int dimension_count);


template
void FindKNearestKdTreePointCloudPoints<open3d::core::Device::DeviceType::CUDA, SearchStrategy::RECURSIVE>(
		open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
		const open3d::core::Tensor& query_points, int32_t k, const void* root, int dimension_count);



void PointCloudDataToHost_CUDA(open3d::core::Blob& node_data_cpu, const open3d::core::Blob& node_data, int point_count, int dimension_count){
	switch (dimension_count) {
		case 1:
			PointCloudDataToHost_CUDA<Eigen::Vector<float, 1>>(node_data_cpu, node_data, point_count);
			break;
		case 2:
			PointCloudDataToHost_CUDA<Eigen::Vector2f>(node_data_cpu, node_data, point_count);
			break;
		case 3:
			PointCloudDataToHost_CUDA<Eigen::Vector3f>(node_data_cpu, node_data, point_count);
			break;
		default:
			PointCloudDataToHost_CUDA<Eigen::Vector<float, Eigen::Dynamic>>(node_data_cpu, node_data, point_count);
			break;
	}
}

} // nnrt::core::kernel::kdtree