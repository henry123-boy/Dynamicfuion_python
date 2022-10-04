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
#pragma once

#include "core/functional/kernel/Sorting.h"
#include "core/functional/kernel/BubbleSort.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/utility/Logging.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	o3c::Device device = unsorted.GetDevice();
	o3c::SizeVector shape = unsorted.GetShape();
	sorted = unsorted.Clone();
	int dimension_count = static_cast<int>(shape.size());
	int64_t stride = shape[dimension_count - 1];
	if(stride > 8){
		utility::LogError("Support for sorting tensor along last dimension where the last dimension is greater than 8 is not supported. "
						  "The last dimension is: {}.", stride);
	}
	int64_t series_count = unsorted.NumElements() / stride;
	TElement* sorted_data = sorted.template GetDataPtr<TElement>();

	o3c::ParallelFor(
			device, series_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
				BubbleSort(series, stride);
#else
				std::sort(series, series + stride);
#endif
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	DISPATCH_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&](){
		SortTensorAlongLastDimension_Dispatched<TDeviceType,scalar_t>(sorted,unsorted);
	});
}

} // namespace nnrt::core::functional::kernel