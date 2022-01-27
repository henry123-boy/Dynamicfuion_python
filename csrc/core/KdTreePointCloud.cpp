//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/11/22.
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
#include <limits>
#include <memory>

#include "core/KdTreePointCloud.h"
#include "core/DeviceSelection.h"
#include "core/kernel/KdTreePointCloud.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;


namespace nnrt::core {
KdTreePointCloud::KdTreePointCloud(const open3d::core::Tensor& points)
		: point_dimension_count(points.GetShape(1)),
		  point_count(points.GetShape(0)),
		  node_data(std::make_shared<open3d::core::Blob>(points.GetLength() * kernel::kdtree::GetNodeByteCount(points), points.GetDevice())),
		  root(nullptr) {
	auto dimensions = points.GetShape();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	if (dimensions.size() != 2) {
		o3u::LogError("KdTreePointCloud can only hold points from two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}
	if (points.GetLength() > std::numeric_limits<int32_t>::max()) {
		o3u::LogError("KdTreePointCloud currently cannot support more than {} points. Got: {} points.", std::numeric_limits<int32_t>::max(),
		              points.GetLength());
	}
	kernel::kdtree::BuildKdTreePointCloud(*this->node_data, points, &this->root);
}


void KdTreePointCloud::FindKNearestToPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
                                            const open3d::core::Tensor& query_points, int32_t k) const {
	o3c::AssertTensorDevice(query_points, this->node_data->GetDevice());
	o3c::AssertTensorDtype(query_points, o3c::Dtype::Float32);
	if (query_points.GetShape().size() != 2 ||
	    query_points.GetShape(1) != point_dimension_count) {
		o3u::LogError("Query point array of shape {} is incompatible to the set of points in the KD Tree Point Cloud, which"
		              "have {} dimensions. Both arrays should be two-dimensional and have matching axis 1 length (i.e. point dimensions).",
		              query_points.GetShape(), point_dimension_count);
	}
	kernel::kdtree::FindKNearestKdTreePointCloudPoints<kernel::kdtree::SearchStrategy::ITERATIVE>(
			nearest_neighbors, nearest_neighbor_distances, query_points, k, this->root, this->point_dimension_count
	);
}

std::string KdTreePointCloud::GenerateTreeDiagram(int digit_length) const {
	std::string diagram;
	if (digit_length % 2 == 0 || digit_length < 1) {
		o3u::LogError("digit_length parameter to `GenerateTreeDiagram` should be odd and greater than one, got {}.",
		              digit_length);
	}
	kernel::kdtree::GenerateKdTreePointCloudDiagram(diagram, *this->node_data, this->root, this->point_count,
													this->point_dimension_count, digit_length);
	return diagram;
}


} // namespace nnrt::core