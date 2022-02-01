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
#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/core/TensorList.h>
#include <open3d/core/Blob.h>
#include "core/PlatformIndependence.h"
#include "core/kernel/KdTreeNodeTypes.h"

namespace nnrt::core {


class KdTree {
public:

	explicit KdTree(const open3d::core::Tensor& points);
	virtual ~KdTree() = default;
	std::string GenerateTreeDiagram(int digit_length = 5) const;
	void FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances,
	                                  const open3d::core::Tensor& query_points, int32_t k, bool sort_output = false) const;

	const kernel::kdtree::KdTreeNode* GetNodes() const;
	int64_t GetNodeCount() const;

private:
	const open3d::core::Tensor& points;

	const int32_t node_count;
	const std::shared_ptr<open3d::core::Blob> nodes;

};

} // namespace nnrt::core