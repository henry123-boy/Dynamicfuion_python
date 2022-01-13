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
#pragma once

#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/core/TensorList.h>
#include <open3d/core/Blob.h>

#include "core/PlatformIndependence.h"

namespace nnrt::core {


class KdTreePointCloud {
public:


	explicit KdTreePointCloud(const open3d::core::Tensor& points);
	virtual ~KdTreePointCloud() = default;
	std::string GenerateTreeDiagram(int digit_length = 5) const;
	virtual void FindKNearestToPoints(open3d::core::Tensor& nearest_neighbors, open3d::core::Tensor& nearest_neighbor_distances,
	                                  const open3d::core::Tensor& query_points, int32_t k) const;


private:
	const int point_dimension_count;
	const int point_count;
	const std::shared_ptr<open3d::core::Blob> node_data;
	void* root;

};

} // namespace nnrt::core