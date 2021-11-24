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
#include "DeviceSelection.h"

#ifdef BUILD_CUDA_MODULE

#include <open3d/core/CUDAUtils.h>

#endif

#include <limits>
#include <cstring>

namespace o3c = open3d::core;
namespace o3u = open3d::utility;


namespace nnrt::core {
KdTree::KdTree(const open3d::core::Tensor& indexed_tensor)
		: indexed_tensor(&indexed_tensor),
		  dimension_count(indexed_tensor.GetShape(1)),
		  index_data(std::make_shared<o3c::Blob>(indexed_tensor.GetShape(0) * sizeof(Node), indexed_tensor.GetDevice())),
		  data_pointer(this->index_data->GetDataPtr()) {
	auto dimensions = indexed_tensor.GetShape();
	if (dimensions.size() != 2) {
		o3u::LogError("KdTree index currently only supports indexing of two-dimensional tensors. "
		              "Provided tensor has dimensions: {}", dimensions);
	}

}

void KdTree::Reindex() {

}

void KdTree::ChangeToAppendedTensor(const open3d::core::Tensor& tensor) {

}

void KdTree::UpdatePoint(const open3d::core::Tensor& point) {

}


} // namespace nnrt::core
