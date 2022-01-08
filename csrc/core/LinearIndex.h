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
#pragma once
// linear index over an array, mostly for benchmarking purposes

#include <open3d/core/Tensor.h>

namespace nnrt::core {


class LinearIndex{
public:


	explicit LinearIndex(const open3d::core::Tensor& points);
	virtual ~LinearIndex() = default;
	virtual void FindKNearestToPoints(open3d::core::Tensor& nearest_neighbor_indices, open3d::core::Tensor& squared_distances, const open3d::core::Tensor& query_points, int32_t k) const;


private:
	const open3d::core::Tensor& points;

};

} // namespace nnrt::core