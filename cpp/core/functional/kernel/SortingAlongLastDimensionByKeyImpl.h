//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/4/22.
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

template<open3d::core::Device::DeviceType TDeviceType, typename TValue, typename TKey, typename TSortByKeyFunction>
void SortTensorAlongLastDimensionByKey_Generic(
		open3d::core::Tensor& sorted_values,
		open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values,
		const open3d::core::Tensor& unsorted_keys,
		TSortByKeyFunction&& sort
) {
	o3c::Device device = unsorted_values.GetDevice();
	o3c::SizeVector shape = unsorted_values.GetShape();
	sorted_values = unsorted_values.Clone();
	sorted_keys = unsorted_keys.Clone();
	int dimension_count = static_cast<int>(shape.size());
	int64_t stride = shape[dimension_count - 1];
	int64_t series_count = unsorted_values.NumElements() / stride;
	TValue* sorted_value_data = sorted_values.template GetDataPtr<TValue>();
	TKey* sorted_key_data = sorted_keys.template GetDataPtr<TKey>();
	sort(device, series_count, sorted_value_data, sorted_key_data, stride);
}

template<typename TKey, typename TValue, typename Compare>
inline void SortByKeyCPU(TKey* key_series, TValue* value_series, int64_t stride, Compare&& compare) {
	std::vector<int> indexes(stride);
	std::iota(indexes.begin(), indexes.end(), 0);
	std::stable_sort(indexes.begin(), indexes.end(), compare);
	std::vector<TValue> sorted_values(stride);
	std::vector<TKey> sorted_keys(stride);
	int i_sorted = 0;
	for (auto i_unsorted: indexes) {
		sorted_values[i_sorted] = value_series[i_unsorted];
		sorted_keys[i_sorted] = key_series[i_unsorted];
		i_sorted++;
	}
	memcpy(value_series, sorted_values.data(), stride * sizeof(TValue));
	memcpy(key_series, sorted_keys.data(), stride * sizeof(TKey));
}

template<open3d::core::Device::DeviceType TDeviceType, typename TValue, typename TKey>
void SortTensorAlongLastDimensionByKey_Dispatched(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys
) {
	SortTensorAlongLastDimensionByKey_Generic<TDeviceType, TValue, TKey>(
			sorted_values, sorted_keys, unsorted_values, unsorted_keys,
			[](o3c::Device device, int64_t series_count, TValue* sorted_value_data, TKey* sorted_key_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TValue* value_series = sorted_value_data + workload_idx * stride;
							TKey* key_series = sorted_key_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSortByKey(value_series, key_series, stride);
#else
							SortByKeyCPU(key_series, value_series, stride, [&key_series](int i0, int i1) {
								return key_series[i0] < key_series[i1];
							});
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TValue, typename TKey>
void SortTensorAlongLastDimensionByKey_PositiveFirst_Dispatched(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys
) {
	SortTensorAlongLastDimensionByKey_Generic<TDeviceType, TValue, TKey>(
			sorted_values, sorted_keys, unsorted_values, unsorted_keys,
			[](o3c::Device device, int64_t series_count, TValue* sorted_value_data, TKey* sorted_key_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TValue* value_series = sorted_value_data + workload_idx * stride;
							TKey* key_series = sorted_key_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSortByKey_PositiveFirst(value_series, key_series, stride);
#else
							SortByKeyCPU(key_series, value_series, stride, [&key_series](int i0, int i1) {
								auto& a = key_series[i0];
								auto& b = key_series[i1];
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

template<open3d::core::Device::DeviceType TDeviceType, typename TValue, typename TKey>
void SortTensorAlongLastDimensionByKey_Descending_Dispatched(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys
) {
	SortTensorAlongLastDimensionByKey_Generic<TDeviceType, TValue, TKey>(
			sorted_values, sorted_keys, unsorted_values, unsorted_keys,
			[](o3c::Device device, int64_t series_count, TValue* sorted_value_data, TKey* sorted_key_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TValue* value_series = sorted_value_data + workload_idx * stride;
							TKey* key_series = sorted_key_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSortByKey_Descending(value_series, key_series, stride);
#else
							SortByKeyCPU(key_series, value_series, stride, [&key_series](int i0, int i1) {
								return key_series[i0] > key_series[i1];
							});
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType, typename TValue, typename TKey>
void SortTensorAlongLastDimensionByKey_Descending_NegativeFirst_Dispatched(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys
) {
	SortTensorAlongLastDimensionByKey_Generic<TDeviceType, TValue, TKey>(
			sorted_values, sorted_keys, unsorted_values, unsorted_keys,
			[](o3c::Device device, int64_t series_count, TValue* sorted_value_data, TKey* sorted_key_data, int64_t stride) {
				o3c::ParallelFor(
						device, series_count,
						NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
							TValue* value_series = sorted_value_data + workload_idx * stride;
							TKey* key_series = sorted_key_data + workload_idx * stride;
#ifdef __CUDACC__
							BubbleSortByKey_Descending_NegativeFirst(value_series, key_series, stride);
#else
							SortByKeyCPU(key_series, value_series, stride, [&key_series](int i0, int i1) {
								auto& a = key_series[i0];
								auto& b = key_series[i1];
								if (b < 0) {
									return a < 0 && a > b;
								} else {
									return a < 0 || a > b;
								}
							});;
#endif
						}
				);
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimensionByKey(
		open3d::core::Tensor& sorted_values, open3d::core::Tensor& sorted_keys,
		const open3d::core::Tensor& unsorted_values, const open3d::core::Tensor& unsorted_keys,
		bool non_negative_first, SortOrder order
) {
	switch (order) {
		case SortOrder::ASC:
			if (non_negative_first) {
				DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(unsorted_keys.GetDtype(), [&]() {
					using key_t = scalar_t;
					DISPATCH_DTYPE_TO_TEMPLATE(unsorted_values.GetDtype(), [&]() {
						using value_t = scalar_t;
						SortTensorAlongLastDimensionByKey_PositiveFirst_Dispatched<TDeviceType, value_t, key_t>(
								sorted_values, sorted_keys, unsorted_values, unsorted_keys);
					});
				});
			} else {
				DISPATCH_DTYPE_TO_TEMPLATE(unsorted_keys.GetDtype(), [&]() {
					using key_t = scalar_t;
					DISPATCH_DTYPE_TO_TEMPLATE(unsorted_values.GetDtype(), [&]() {
						using value_t = scalar_t;
						SortTensorAlongLastDimensionByKey_Dispatched<TDeviceType, value_t, key_t>(
								sorted_values, sorted_keys, unsorted_values, unsorted_keys);
					});
				});
			}
			break;
		case SortOrder::DESC:
			if (non_negative_first) {
				DISPATCH_DTYPE_TO_TEMPLATE(unsorted_keys.GetDtype(), [&]() {
					using key_t = scalar_t;
					DISPATCH_DTYPE_TO_TEMPLATE(unsorted_values.GetDtype(), [&]() {
						using value_t = scalar_t;
						SortTensorAlongLastDimensionByKey_Descending_Dispatched<TDeviceType, value_t, key_t>(
								sorted_values, sorted_keys, unsorted_values, unsorted_keys);
					});

				});
			} else {
				DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(unsorted_keys.GetDtype(), [&]() {
					using key_t = scalar_t;
					DISPATCH_DTYPE_TO_TEMPLATE(unsorted_values.GetDtype(), [&]() {
						using value_t = scalar_t;
						SortTensorAlongLastDimensionByKey_Descending_NegativeFirst_Dispatched<TDeviceType, value_t, key_t>(
								sorted_values, sorted_keys, unsorted_values, unsorted_keys);
					});
				});
			}
			break;
		default: utility::LogError("Unsupported sort order constant: {}", order);
	}
}


} // namespace nnrt::core::functional::kernel