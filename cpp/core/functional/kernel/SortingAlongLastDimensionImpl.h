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
// stdlib
#include <numeric>
// 3rd party
#include <open3d/core/ParallelFor.h>
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/Dispatch.h>
#include <open3d/utility/Logging.h>
#include <Eigen/Dense>
// local
#include "core/functional/kernel/Sorting.h"
#include "core/functional/kernel/BubbleSort.h"
#include "core/Dispatch.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {


template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TSortFunction>
void SortTensorAlongLastDimension_Generic(
		open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted,
		TSortFunction&& sort
) {
	o3c::Device device = unsorted.GetDevice();
	o3c::SizeVector shape = unsorted.GetShape();
	sorted = unsorted.Clone();
	int dimension_count = static_cast<int>(shape.size());
	int64_t stride = shape[dimension_count - 1];
	int64_t series_count = unsorted.NumElements() / stride;
	TElement* sorted_data = sorted.template GetDataPtr<TElement>();
	sort(device, series_count, sorted_data, stride);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	SortTensorAlongLastDimension_Generic<TDeviceType, TElement>(
			sorted, unsorted,
			[](o3c::Device device, int64_t series_count, TElement* sorted_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSort(series, stride);
#else
							std::sort(series, series + stride);
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_PositiveFirst_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	SortTensorAlongLastDimension_Generic<TDeviceType, TElement>(
			sorted, unsorted,
			[](o3c::Device device, int64_t series_count, TElement* sorted_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSort_PositiveFirst(series, stride);
#else
							std::sort(series, series + stride, [](const TElement& a, const TElement& b) {
								if (b >= 0) {
									return a > 0 && a < b;
								} else {
									return a > 0 || a < b;
								}
							});
#endif
						}
				);
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_Descending_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	SortTensorAlongLastDimension_Generic<TDeviceType, TElement>(
			sorted, unsorted,
			[](o3c::Device device, int64_t series_count, TElement* sorted_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSort_Descending(series, stride);
#else
							std::sort(series, series + stride, [](const TElement& a, const TElement& b) {
								return a > b;
							});
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_Descending_NegativeFirst_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	SortTensorAlongLastDimension_Generic<TDeviceType, TElement>(
			sorted, unsorted,
			[](o3c::Device device, int64_t series_count, TElement* sorted_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSort_Descending_NegativeFirst(series, stride);
#else
							std::sort(series, series + stride, [](const TElement& a, const TElement& b) {
								if (b < 0) {
									return a < 0 && a > b;
								} else {
									return a < 0 || a > b;
								}
							});
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool non_negative_first, SortOrder order) {
	switch (order) {
		case SortOrder::ASC:
			if (non_negative_first) {
				DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
					SortTensorAlongLastDimension_PositiveFirst_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
				});
			} else {
				DISPATCH_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
					SortTensorAlongLastDimension_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
				});
			}
			break;
		case SortOrder::DESC:
			if (non_negative_first) {
				DISPATCH_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
					SortTensorAlongLastDimension_Descending_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
				});
			} else {
				DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
					SortTensorAlongLastDimension_Descending_NegativeFirst_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
				});
			}
			break;
		default: utility::LogError("Unsupported sort order constant: {}", order);
	}

}

} // namespace nnrt::core::functional::kernel