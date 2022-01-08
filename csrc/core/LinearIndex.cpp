//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/8/22.
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
#include "LinearIndex.h"
#include <limits>

#include "core/kernel/LinearIndex.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace nnrt::core {
nnrt::core::LinearIndex::LinearIndex(const open3d::core::Tensor& points) :
	points(points){
	auto dimensions = points.GetShape();
	points.AssertDtype(o3c::Dtype::Float32);
	if (dimensions.size() != 2) {
		o3u::LogError("Linear index only supports indexing of two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}
	if(points.GetLength() > std::numeric_limits<int32_t>::max()){
		o3u::LogError("Linear index doesn't support more than {} points. Got: {} points.", std::numeric_limits<int32_t>::max(),
		              points.GetLength());
	}
}

void LinearIndex::FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
                                       const open3d::core::Tensor& query_points, int32_t k) const {
	query_points.AssertDevice(this->points.GetDevice());
	query_points.AssertDtype(o3c::Dtype::Float32);
	if(query_points.GetShape().size() != 2 ||
	   query_points.GetShape(1) != this->points.GetShape(1)){
		o3u::LogError("Reference point array of shape {} is incompatible to the set of points being indexed by the linear index, which"
		              "has shape {}. Both arrays should be two-dimensional and have matching axis 1 length (i.e. point dimensions).",
		              query_points.GetShape(), this->points.GetShape());
	}
	kernel::linear_index::FindKNearestKdTreePoints(nearest_neighbor_indices, squared_distances, query_points, k, this->points);
}

} // nnrt::core
