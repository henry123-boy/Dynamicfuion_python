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

open3d::core::Tensor SortTensorAlongLastDimension(const open3d::core::Tensor& unsorted) {
	if (unsorted.NumDims() == 0 || unsorted.NumElements() == 0) {
		return unsorted;
	}
	o3c::Tensor sorted;
	kernel::SortTensorAlongLastDimension(sorted, unsorted.Contiguous());
	return sorted;
}

} // nnrt::core::functional