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
#include "core/functional/kernel/Comparisons.h"
#include "core/DeviceSelection.h"

namespace nnrt::core::functional::kernel {

void LastDimensionSeriesMatchUpToNElements(open3d::core::Tensor& matches, const open3d::core::Tensor& tensor_a, const open3d::core::Tensor& tensor_b,
                                           int32_t max_mismatches_per_series, double rtol, double atol) {
	ExecuteOnDevice(
			tensor_a.GetDevice(),
			[&]() {
				LastDimensionSeriesMatchUpToNElements<open3d::core::Device::DeviceType::CPU>(
						matches, tensor_a, tensor_b, max_mismatches_per_series, rtol, atol
				);
			},
			[&]() {
				NNRT_IF_CUDA(LastDimensionSeriesMatchUpToNElements<open3d::core::Device::DeviceType::CUDA>(
						matches, tensor_a, tensor_b, max_mismatches_per_series, rtol, atol
				););
			}
	);
}

} // namespace nnrt::core::functional::kernel