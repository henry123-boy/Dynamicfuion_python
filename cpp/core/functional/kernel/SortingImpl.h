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

// 3rd party
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

// local
#include "core/functional/kernel/Sorting.h"
#include "core/functional/kernel/BubbleSort.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::functional::kernel {


template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_PositiveFirst_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	o3c::Device device = unsorted.GetDevice();
	o3c::SizeVector shape = unsorted.GetShape();
	sorted = unsorted.Clone();
	int dimension_count = static_cast<int>(shape.size());
	int64_t stride = shape[dimension_count - 1];
	int64_t series_count = unsorted.NumElements() / stride;
	TElement* sorted_data = sorted.template GetDataPtr<TElement>();

	o3c::ParallelFor(
			device, series_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				TElement* series = sorted_data + workload_idx * stride;
#ifdef __CUDACC__
				BubbleSort_PositiveFirst(series, stride);
#else
				std::sort(series, series + stride, [](TElement a, TElement b) {
					if (b >= 0) {
						return a < 0 || a > b;
					} else {
						return a < 0 && a > b;
					}
				});
#endif
			}
	);

}

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
void SortTensorAlongLastDimension_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	o3c::Device device = unsorted.GetDevice();
	o3c::SizeVector shape = unsorted.GetShape();
	sorted = unsorted.Clone();
	int dimension_count = static_cast<int>(shape.size());
	int64_t stride = shape[dimension_count - 1];
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


#define DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(DTYPE, ...)                   \
    [&] {                                                        \
        if (DTYPE == open3d::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int8) {                \
            using scalar_t = int8_t;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int16) {               \
            using scalar_t = int16_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int32) {               \
            using scalar_t = int32_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == open3d::core::Int64) {               \
            using scalar_t = int64_t;                            \
            return __VA_ARGS__();                                \
        } else {                                                 \
            open3d::utility::LogError("Unsupported data type."); \
        }                                                        \
    }()


template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, bool positive_first) {
	if (positive_first) {
		DISPATCH_SIGNED_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
			SortTensorAlongLastDimension_PositiveFirst_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
		});
	} else {
		DISPATCH_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
			SortTensorAlongLastDimension_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
		});
	}
}

#define DISPATCH_VECTOR_2_to_4_SIZE_TO_EIGEN_TYPE(SIZE, ELEMENT_TYPE, ...) \
    [&]{                                                            \
        switch(SIZE){                                               \
        case 2:{                                                    \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 2>;        \
            return __VA_ARGS__();                                   \
            }                                                       \
        case 3:{                                                    \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 3>;        \
            return __VA_ARGS__();                                   \
            }                                                       \
        case 4:{                                                    \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 4>;        \
            return __VA_ARGS__();                                   \
            }                                                       \
        default:                                                    \
            open3d::utility::LogError("Unsupported size, {}."       \
            " Only sizes 2-4 are supported.", SIZE);                \
            return;                                                 \
        }                                                           \
    }()


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
	thrust::sort(thrust::device, data_start, data_end, [column] __device__ (const TRow& a, const TRow& b){
		return a(column) < b(column);
	});
#else
	std::sort(data_start, data_end, [&column](const TRow& a, const TRow& b) {
		return a.coeff(column) < b.coeff(column);
	});
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column, bool in_place) {
	int64_t row_count = unsorted.GetLength();
	int64_t column_count = unsorted.GetShape(1);
	o3c::AssertTensorShape(unsorted, { row_count, column_count });
	o3c::AssertTensorDtypes(unsorted, { o3c::Float32, o3c::Float64, o3c::Int32 });
	if (column < 0 || column >= column_count) {
		utility::LogError("Column index ({}) must be a non-zero value below column_count ({}).", column, column_count);
	}
	//TODO a more-versatile d-type dispatching macro
	if (unsorted.GetDtype() == o3c::Int32) {
		DISPATCH_VECTOR_2_to_4_SIZE_TO_EIGEN_TYPE(column_count, int32_t, [&]() {
			SortTensorByColumn_Dispatched<TDeviceType, int32_t, vector_t>(sorted, unsorted, column);
		});
	} else {
		DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
				unsorted.GetDtype(),
				[&]() {
					DISPATCH_VECTOR_2_to_4_SIZE_TO_EIGEN_TYPE(column_count, scalar_t, [&]() {
						SortTensorByColumn_Dispatched<TDeviceType, scalar_t, vector_t>(sorted, unsorted, column);
					});
				}
		);
	}
}

} // namespace nnrt::core::functional::kernel