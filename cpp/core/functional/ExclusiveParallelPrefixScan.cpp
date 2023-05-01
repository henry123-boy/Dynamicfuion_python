//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/25/23.
//  Copyright (c) 2023 Gregory Kramida
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
// stdlib includes

// third-party includes

// local includes
#include "core/functional/ExclusiveParallelPrefixScan.h"
#include "core/functional/kernel/ExclusiveParallelPrefixScan.h"

namespace o3c = open3d::core;

namespace nnrt::core::functional {

open3d::core::Tensor ExclusiveParallelPrefixSum1D(const open3d::core::Tensor& source) {
	o3c::Tensor prefix_sum;
	kernel::ExclusiveParallelPrefixSum1D(prefix_sum, source);
	return prefix_sum;
}

} // namespace nnrt::core::functional