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

namespace nnrt::core {


class KdTree{
public:


	explicit KdTree(const open3d::core::Tensor& points);
	virtual ~KdTree() = default;
	virtual void Reindex();
	virtual void UpdatePoint(const open3d::core::Tensor& point);
	virtual void ChangeToAppendedTensor(const open3d::core::Tensor& tensor);
	virtual open3d::core::TensorList FindKNearestToPoints(const open3d::core::Tensor& query_points, int32_t k) const;


private:
	const open3d::core::Tensor& points;
	const int64_t point_dimension_count;

	const std::shared_ptr<open3d::core::Blob> index_data;
	void* index_data_pointer;

};

} // namespace nnrt::core