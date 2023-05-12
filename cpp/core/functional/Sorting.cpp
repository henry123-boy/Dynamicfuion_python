//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/4/22.
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
// local
#include "core/functional/Sorting.h"
#include "core/functional/kernel/Sorting.h"

namespace o3c = open3d::core;

namespace nnrt::core::functional {

open3d::core::Tensor SortTensorAlongLastDimension(const open3d::core::Tensor& unsorted, bool positive_first) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted;
	kernel::SortTensorAlongLastDimension(sorted, unsorted.Contiguous(), positive_first);
	return sorted;
}

open3d::core::Tensor SortTensorByColumn(const open3d::core::Tensor& unsorted, int column) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted;
	kernel::SortTensorByColumn(sorted, unsorted, column, false);
	return sorted;
}

open3d::core::Tensor SortTensorByColumns(const open3d::core::Tensor& unsorted, const o3c::SizeVector& columns) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted = unsorted.Clone();
	for(const int64_t& column : columns){
		kernel::SortTensorByColumn(sorted, sorted, static_cast<int32_t>(column), true);
	}
	return sorted;
}

} // nnrt::core::functional