//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/24/23.
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
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/Dispatch.h>
#include <open3d/utility/Logging.h>
#include <Eigen/Dense>
#ifdef __CUDACC__
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#endif

// local includes
#include "core/functional/kernel/Sorting.h"
#include "core/Dispatch.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {


template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TRow>
void SortTensorByColumn_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place = false) {
	if (in_place) {
		sorted = unsorted;
	} else {
		sorted = unsorted.Clone();
	}

	TRow* data_start = reinterpret_cast<TRow*>(sorted.GetDataPtr());
	TRow* data_end = data_start + sorted.GetLength();
#ifdef __CUDACC__
	o3c::Device device = unsorted.GetDevice();
	cudaSetDevice(device.GetID());
	thrust::stable_sort(thrust::device, data_start, data_end, [column] __device__ (const TRow& a, const TRow& b){
		return a(column) < b(column);
	});
#else
	std::stable_sort(data_start, data_end, [&column](const TRow& a, const TRow& b) {
		return a.coeff(column) < b.coeff(column);
	});
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place) {
	int64_t row_count = unsorted.GetLength();
	int64_t column_count = unsorted.GetShape(1);
	o3c::AssertTensorShape(unsorted, { row_count, column_count });
	o3c::AssertTensorDtypes(unsorted, { o3c::Float32, o3c::Float64, o3c::Int32, o3c::Int64 });
	if (column < 0 || column >= column_count) {
		utility::LogError("Column index ({}) must be a non-zero value below column_count ({}).", column, column_count);
	}
	//TODO a more-versatile d-type dispatching macro
	if (unsorted.GetDtype() == o3c::Int32 || unsorted.GetDtype() == o3c::Int64) {
		DISPATCH_SIGNED_ONE_OR_TWO_WORD_DTYPE_TO_TEMPLATE(
				unsorted.GetDtype(),
				[&]() {
					DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE(column_count, scalar_t , [&]() {
						SortTensorByColumn_Dispatched<TDeviceType, scalar_t, vector_t>(sorted, unsorted, column, in_place);
					});
				}
		);
	} else {
		DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
				unsorted.GetDtype(),
				[&]() {
					DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE(column_count, scalar_t, [&]() {
						SortTensorByColumn_Dispatched<TDeviceType, scalar_t, vector_t>(sorted, unsorted, column, in_place);
					});
				}
		);
	}
}

template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TRow>
void ArgSortTensorByColumn_Dispatched(open3d::core::Tensor& index, const open3d::core::Tensor& unsorted, int column) {
	index = o3c::Tensor::Arange(0, unsorted.GetLength(), 1, o3c::Int64, unsorted.GetDevice());
	o3c::Tensor sorted = unsorted.Clone();

	TRow* data_start = reinterpret_cast<TRow*>(sorted.GetDataPtr());
	auto index_start = index.GetDataPtr<int64_t>();
#ifdef __CUDACC__
	TRow* data_end = data_start + sorted.GetLength();
	o3c::Device device = unsorted.GetDevice();
	cudaSetDevice(device.GetID());
	thrust::sort_by_key(thrust::device, data_start, data_end, index_start, [column] __device__ (const TRow& a, const TRow& b){
		return a(column) < b(column);
	});
#else
	auto index_end = index_start + index.GetLength();
	std::sort(index_start, index_end, [&column, &data_start](const int64_t& a, const int64_t& b) {
		return data_start[a].coeff(column) < data_start[b].coeff(column);
	});
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void ArgSortTensorByColumn(open3d::core::Tensor& index, const open3d::core::Tensor& unsorted, int column) {
	int64_t row_count = unsorted.GetLength();
	int64_t column_count = unsorted.GetShape(1);
	o3c::AssertTensorShape(unsorted, { row_count, column_count });
	o3c::AssertTensorDtypes(unsorted, { o3c::Float32, o3c::Float64, o3c::Int32 });
	if (column < 0 || column >= column_count) {
		utility::LogError("Column index ({}) must be a non-zero value below column_count ({}).", column, column_count);
	}
	//TODO a more-versatile d-type dispatching macro
	if (unsorted.GetDtype() == o3c::Int32) {
		DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE(column_count, int32_t, [&]() {
			ArgSortTensorByColumn_Dispatched<TDeviceType, int32_t, vector_t>(index, unsorted, column);
		});
	} else {
		DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
				unsorted.GetDtype(),
				[&]() {
					DISPATCH_VECTOR_1_to_4_SIZE_TO_EIGEN_TYPE(column_count, scalar_t, [&]() {
						ArgSortTensorByColumn_Dispatched<TDeviceType, scalar_t, vector_t>(index, unsorted, column);
					});
				}
		);
	}
}

} // namespace nnrt::core::functional::kernel