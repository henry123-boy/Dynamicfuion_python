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
#pragma once

//3rd party
#include "open3d/core/ParallelFor.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/utility/Logging.h"

#ifndef __CUDACC__
#include <atomic>
#endif

//local
#include "core/functional/kernel/Comparisons.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/platform_independence/Atomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void LastDimensionSeriesMatchUpToNElements_Dispatched(
		open3d::core::Tensor& matches, const open3d::core::Tensor& tensor_a, const open3d::core::Tensor& tensor_b,
		int32_t max_mismatches_per_series, double rtol, double atol
) {
	o3c::Device device = tensor_a.GetDevice();
	o3c::SizeVector shape = tensor_a.GetShape();
	int dimension_count = static_cast<int>(shape.size());
	int64_t series_length = shape[dimension_count - 1];
	int64_t series_count = tensor_a.NumElements() / series_length;
	const TElement* data_a = tensor_a.template GetDataPtr<TElement>();
	const TElement* data_b = tensor_b.template GetDataPtr<TElement>();
	o3c::Tensor series_match_counts = o3c::Tensor::Zeros({series_count}, o3c::Int32, device);
	int32_t* series_match_counts_data = series_match_counts.GetDataPtr<int32_t>();

#ifdef __CUDACC__
	o3c::Tensor tensor_b_matched_mask = o3c::Tensor::Zeros(shape, o3c::Bool, device);
	bool* tensor_b_matched_mask_data = tensor_b_matched_mask.template GetDataPtr<bool>();
#else
	std::vector<std::atomic_bool> tensor_b_matched_mask(tensor_b.NumElements());
	std::vector<std::atomic_int> series_match_count_atomics(series_count);
	o3c::ParallelFor(
			device, tensor_b.NumElements(),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_series = workload_idx / series_length;
				series_match_count_atomics[i_series].store(0);
				tensor_b_matched_mask[workload_idx].store(false);
			}
	);
#endif

	o3c::ParallelFor(
			device, tensor_a.NumElements(),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_series = workload_idx / series_length;
				int64_t i_element_a_within_series = workload_idx % series_length;
				const TElement& element_a = data_a[workload_idx];
				for (int i_element_b_within_series = 0; i_element_b_within_series < series_length; i_element_b_within_series++) {

					int64_t i_element_b = workload_idx - i_element_a_within_series + i_element_b_within_series;
					const TElement& element_b = data_b[i_element_b];
					auto actual_error = static_cast<TElement>(abs(static_cast<double>(element_a - element_b)));
					TElement max_error = atol + rtol * static_cast<TElement>(abs(static_cast<double>(element_b)));
					bool matches = actual_error <= max_error;

					if (matches) {
#ifdef __CUDACC__
						bool* match_atomic = tensor_b_matched_mask_data + i_element_b;
#else
						std::atomic<bool>& match_atomic = tensor_b_matched_mask[i_element_b];


#endif
						if (NNRT_ATOMIC_CE(match_atomic, false, true)) {
#ifdef __CUDACC__
							int* count_atomic = series_match_counts_data + i_series;
#else
							std::atomic_int& count_atomic = series_match_count_atomics[i_series];
#endif
							NNRT_ATOMIC_ADD(count_atomic, 1);
						}
					}
				}
			}
	);

#ifndef __CUDACC__
	o3c::ParallelFor(
			device, series_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				series_match_counts_data[workload_idx] = series_match_count_atomics[workload_idx].load();
			}
	);
#endif
	matches = series_match_counts >= o3c::Tensor::Full({series_count}, series_length - max_mismatches_per_series, o3c::Int32, device);
}

template<open3d::core::Device::DeviceType TDeviceType>
void LastDimensionSeriesMatchUpToNElements(
		open3d::core::Tensor& matches, const open3d::core::Tensor& tensor_a, const open3d::core::Tensor& tensor_b,
		int32_t max_mismatches_per_series, double rtol, double atol
) {
	DISPATCH_DTYPE_TO_TEMPLATE(tensor_a.GetDtype(), [&]() {
		LastDimensionSeriesMatchUpToNElements_Dispatched<TDeviceType, scalar_t>(
				matches, tensor_a, tensor_b, max_mismatches_per_series, rtol, atol
		);
	});
}

} // namespace nnrt::core::functional::kernel
