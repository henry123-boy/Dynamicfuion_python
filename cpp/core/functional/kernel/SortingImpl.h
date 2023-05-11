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


template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorAlongLastDimension(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted) {
	DISPATCH_DTYPE_TO_TEMPLATE(unsorted.GetDtype(), [&]() {
		SortTensorAlongLastDimension_Dispatched<TDeviceType, scalar_t>(sorted, unsorted);
	});
}

#define DISPATCH_VECTOR_SIZE_TO_EIGEN_TYPE(SIZE, ELEMENT_TYPE, ...) \
    [&]{                                                            \
        switch(SIZE){                                               \
        case 1:{                                                    \
            using vector_t =  Eigen::Vector<ELEMENT_TYPE, 1>;       \
            return __VA_ARGS__();                                   \
            }                                                       \
        case 2:{                                                    \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 2>;        \
            return __VA_ARGS__();                                   \
            }                                                       \
        case 3:{                                                    \
            using vector_t = Eigen::Vector<ELEMENT_TYPE, 3>;        \
            return __VA_ARGS__();                                   \
            }                                                       \
        default:                                                    \
            open3d::utility::LogError("Unsupported size, {}."       \
			" Only sizes 1-8 are supported.",SIZE);                 \
            return;                                                 \
        }                                                           \
    }()


template<open3d::core::Device::DeviceType TDeviceType, typename TElement, typename TRow>
void SortTensorByColumn_Dispatched(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column) {
	sorted = unsorted.Clone();
	TRow* data_start = reinterpret_cast<TRow*>(sorted.GetDataPtr());
	TRow* data_end = data_start + sorted.GetLength();
#ifdef __CUDACC__
	o3c::Device device = unsorted.GetDevice();
	cudaSetDevice(device.GetID());
	thrust::sort(thrust::device, data_start, data_end, [column] __device__ (const TRow& a, const TRow& b){
		return a(column) < b(column);
	});
#else
	std::sort(data_start, data_end, [&column](const TRow& a, const TRow& b){
		return a.coeff(column) < b.coeff(column);
	});
#endif
}

template<open3d::core::Device::DeviceType TDeviceType>
void SortTensorByColumn(open3d::core::Tensor& sorted, const open3d::core::Tensor& unsorted, int column) {
	int64_t row_count = unsorted.GetLength();
	int64_t column_count = unsorted.GetShape(1);
	o3c::AssertTensorShape(unsorted, { row_count, column_count });
	o3c::AssertTensorDtypes(unsorted, {o3c::Float32, o3c::Float64});
	if (column >= column_count) {
		utility::LogError("Column index ({}) must be below column_count ({}).", column, column_count);
	}

	DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(
			unsorted.GetDtype(),
			[&]() {
				DISPATCH_VECTOR_SIZE_TO_EIGEN_TYPE(column_count, scalar_t, [&]() {
					SortTensorByColumn_Dispatched<TDeviceType, scalar_t, vector_t>(sorted, unsorted, column);
				});
			}
	);
}

} // namespace nnrt::core::functional::kernel