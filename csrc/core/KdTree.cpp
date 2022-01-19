//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/24/21.
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

#include "core/KdTree.h"
#include "core/DeviceSelection.h"
#include "core/kernel/KdTreeUtils.h"
#include "core/kernel/KdTree.h"
#include "core/kernel/KdTreeNodeTypes.h"

#include <limits>

namespace o3c = open3d::core;
namespace o3u = open3d::utility;


namespace nnrt::core {
KdTree::KdTree(const open3d::core::Tensor& points)
		: points(points),
		  index_length(core::kernel::kdtree::FindBalancedTreeIndexLength(points.GetLength())),
		  index_data(std::make_shared<open3d::core::Blob>(index_length * sizeof(kernel::kdtree::KdTreeNode), points.GetDevice())) {
	auto dimensions = points.GetShape();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	if (dimensions.size() != 2) {
		o3u::LogError("KdTree index currently only supports indexing of two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}
	if (points.GetLength() > std::numeric_limits<int32_t>::max() || points.GetLength() < 1) {
		o3u::LogError("KdTree index currently cannot support less than 1 or more than {} points. Got: {} points.", std::numeric_limits<int32_t>::max(),
		              points.GetLength());
	}
	kernel::kdtree::BuildKdTreeIndex(*this->index_data, this->index_length, this->points);
}



void KdTree::FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                                  const open3d::core::Tensor& query_points, int32_t k, bool sort_output) const {
	o3c::AssertTensorDevice(query_points, this->points.GetDevice());
	o3c::AssertTensorDtype(query_points, o3c::Dtype::Float32);
	if (query_points.GetShape().size() != 2 ||
	    query_points.GetShape(1) != this->points.GetShape(1)) {
		o3u::LogError("Query point array of shape {} is incompatible to the set of points being indexed by the KD Tree (reference points), which"
		              "has shape {}. Both arrays should be two-dimensional and have matching axis 1 length (i.e. point dimensions).",
		              query_points.GetShape(), this->points.GetShape());
	}
	if (sort_output) {
		kernel::kdtree::FindKNearestKdTreePoints<kernel::kdtree::SearchStrategy::ITERATIVE, kernel::kdtree::NeighborTrackingStrategy::PRIORITY_QUEUE>(
				*this->index_data, this->index_length, nearest_neighbor_indices, squared_distances, query_points, k, this->points);
	} else {
		kernel::kdtree::FindKNearestKdTreePoints<kernel::kdtree::SearchStrategy::ITERATIVE, kernel::kdtree::NeighborTrackingStrategy::PLAIN>(
				*this->index_data,  this->index_length, nearest_neighbor_indices, squared_distances, query_points, k, this->points);
	}

}

std::string KdTree::GenerateTreeDiagram(int digit_length) const {
	std::string diagram;
	if (digit_length % 2 == 0 || digit_length < 1) {
		o3u::LogError("digit_length parameter to `GenerateTreeDiagram` should be odd and greater than one, got {}.",
		              digit_length);
	}
	kernel::kdtree::GenerateTreeDiagram(diagram, *this->index_data, this->index_length, this->points, digit_length);
	return diagram;
}


} // namespace nnrt::core
