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
#include "core/kernel/KdTree.h"

#include <limits>

namespace o3c = open3d::core;
namespace o3u = open3d::utility;


namespace nnrt::core {
KdTree::KdTree(const open3d::core::Tensor& points)
		: points(points),
		  index_data(std::make_shared<open3d::core::Blob>(points.GetLength() * sizeof(kernel::kdtree::KdTreeNode), points.GetDevice())),
		  root(nullptr){
	auto dimensions = points.GetShape();
	points.AssertDtype(o3c::Dtype::Float32);
	if (dimensions.size() != 2) {
		o3u::LogError("KdTree index currently only supports indexing of two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}
	if(points.GetLength() > std::numeric_limits<int32_t>::max()){
		o3u::LogError("KdTree index currently cannot support more than {} points. Got: {} points.", std::numeric_limits<int32_t>::max(),
					  points.GetLength());
	}
	kernel::kdtree::BuildKdTreeIndex(*this->index_data, this->points, &this->root);
}

void KdTree::Reindex() {
	kernel::kdtree::BuildKdTreeIndex(*this->index_data, this->points, &this->root);
}

void KdTree::ChangeToAppendedTensor(const open3d::core::Tensor& tensor) {
	o3u::LogError("Not implemented");
}

void KdTree::UpdatePoint(const open3d::core::Tensor& point) {
	o3u::LogError("Not implemented");
}

void KdTree::FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points, int32_t k) const{
	query_points.AssertDevice(this->points.GetDevice());
	query_points.AssertDtype(o3c::Dtype::Float32);
	if(query_points.GetShape().size() != 2 ||
	   query_points.GetShape(1) != this->points.GetShape(1)){
		o3u::LogError("Reference point array of shape {} is incompatible to the set of points being indexed by the KD Tree, which"
					  "has shape {}. Both arrays should be two-dimensional and have matching axis 1 length (i.e. point dimensions).",
		              query_points.GetShape(), this->points.GetShape());
	}
	kernel::kdtree::FindKNearestKdTreePoints(nearest_neighbor_indices, squared_distances, query_points, k, *this->index_data, this->points, this->root);
}

std::string KdTree::GenerateTreeDiagram(int digit_length) const{
	std::string diagram;
	if(digit_length % 2 == 0 || digit_length < 1){
		o3u::LogError("digit_length parameter to `GenerateTreeDiagram` should be odd and greater than one, got {}.",
		              digit_length);
	}
	kernel::kdtree::GenerateTreeDiagram(diagram, *this->index_data, this->root, this->points, digit_length);
	return diagram;
}


} // namespace nnrt::core
