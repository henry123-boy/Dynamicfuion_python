//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/5/22.
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
#include "core/functional/Comparisons.h"
#include "core/functional/kernel/Comparisons.h"

namespace o3c = open3d::core;
namespace nnrt::core::functional {

open3d::core::Tensor LastDimensionSeriesMatchUpToNElements(
		const open3d::core::Tensor& tensor_a, const open3d::core::Tensor& tensor_b,
		int32_t max_mismatches_per_series, double rtol/*= 1e-5*/, double atol/*= 1e-8*/
) {
	o3c::AssertTensorDevice(tensor_b, tensor_a.GetDevice());
	o3c::AssertTensorDtype(tensor_b, tensor_a.GetDtype());
	o3c::AssertTensorShape(tensor_b, tensor_a.GetShape());

	if (tensor_a.NumDims() == 0 || tensor_a.NumElements() == 0) {
		return o3c::Tensor({0}, o3c::Bool, tensor_a.GetDevice());
	}

	o3c::Tensor matches;
	kernel::LastDimensionSeriesMatchUpToNElements(matches, tensor_a.Contiguous(), tensor_b.Contiguous(), max_mismatches_per_series, rtol, atol);

	return matches;
}

} // namespace nnrt::core::functional