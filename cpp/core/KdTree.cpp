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
#include "core/kernel/KdTreeUtilities.h"
#include "core/kernel/KdTree.h"
#include "core/kernel/KdTreeNodeTypes.h"
#include "open3d/core/MemoryManager.h"
#include "core/functional/Sorting.h"

#include <limits>
#include <utility>

namespace o3c = open3d::core;
namespace utility = open3d::utility;


namespace nnrt::core {
KdTree::KdTree(const open3d::core::Tensor& points)
		: points(points),
		  node_count(core::kernel::kdtree::FindBalancedTreeIndexLength((int) points.GetLength())),
		  nodes(std::make_shared<open3d::core::Blob>(node_count * sizeof(kernel::kdtree::KdTreeNode), points.GetDevice())) {
	auto dimensions = points.GetShape();
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	if (dimensions.size() != 2) {
		utility::LogError("KdTree index currently only supports indexing of two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}
	if (points.GetLength() > std::numeric_limits<int32_t>::max() || points.GetLength() < 1) {
		utility::LogError("KdTree index currently cannot support less than 1 or more than {} points. Got: {} points.",
		                  std::numeric_limits<int32_t>::max(),
		                  points.GetLength());
	}
	kernel::kdtree::BuildKdTreeIndex(*this->nodes, this->node_count, this->points);
}


void KdTree::FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                                  const open3d::core::Tensor& query_points, int32_t k, bool sort_output) const {
	o3c::AssertTensorDevice(query_points, this->points.GetDevice());
	o3c::AssertTensorDtype(query_points, o3c::Dtype::Float32);
	if (query_points.GetShape().size() != 2 ||
	    query_points.GetShape(1) != this->points.GetShape(1)) {
		utility::LogError("Query point array of shape {} is incompatible to the set of points being indexed by the KD Tree (reference points), which"
		              "has shape {}. Both arrays should be two-dimensional and have matching axis 1 length (i.e. point dimensions).",
		                  query_points.GetShape(), this->points.GetShape());
	}
	if (sort_output) {
		if (k > 20) {
			kernel::kdtree::FindKNearestKdTreePoints<kernel::kdtree::NeighborTrackingStrategy::PRIORITY_QUEUE>(
					*this->nodes, this->node_count, nearest_neighbor_indices, squared_distances, query_points, k, this->points);
		} else {
			kernel::kdtree::FindKNearestKdTreePoints<kernel::kdtree::NeighborTrackingStrategy::PLAIN>(
					*this->nodes, this->node_count, nearest_neighbor_indices, squared_distances, query_points, k, this->points);
			//TODO: this is wrong, sort by distance, not index!
			nearest_neighbor_indices = core::functional::SortTensorAlongLastDimension(nearest_neighbor_indices, true, core::functional::SortOrder::ASC);
		}
	} else {
		kernel::kdtree::FindKNearestKdTreePoints<kernel::kdtree::NeighborTrackingStrategy::PLAIN>(
				*this->nodes, this->node_count, nearest_neighbor_indices, squared_distances, query_points, k, this->points);
	}

}

std::string KdTree::GenerateTreeDiagram(int digit_length) const {
	std::string diagram;
	if (digit_length % 2 == 0 || digit_length < 1) {
		utility::LogError("digit_length parameter to `GenerateTreeDiagram` should be odd and greater than one, got {}.",
		                  digit_length);
	}
	kernel::kdtree::GenerateTreeDiagram(diagram, *this->nodes, this->node_count, this->points, digit_length);
	return diagram;
}

const kernel::kdtree::KdTreeNode* KdTree::GetNodes() const {
	return reinterpret_cast<const kernel::kdtree::KdTreeNode*>(this->nodes->GetDataPtr());
}

int32_t KdTree::GetNodeCount() const {
	return this->node_count;
}

KdTree KdTree::Clone() const {
	o3c::Tensor points_copy = points.Clone();
	auto nodes_total_byte_size = node_count * sizeof(kernel::kdtree::KdTreeNode);
	auto nodes_copy = std::make_shared<open3d::core::Blob>(nodes_total_byte_size, points.GetDevice());
	auto device = this->points.GetDevice();
	o3c::MemoryManager::Memcpy(nodes_copy->GetDataPtr(), device,
	                           this->nodes->GetDataPtr(), device,
	                           nodes_total_byte_size);
	return {this->points, this->node_count, nodes_copy};
}

KdTree::KdTree(open3d::core::Tensor points, int32_t node_count, std::shared_ptr<open3d::core::Blob> nodes) :
		points(std::move(points)), node_count(node_count),
		nodes(std::move(nodes)) {}

open3d::core::Tensor KdTree::GetPoints() const {
	return this->points;
}

} // namespace nnrt::core
