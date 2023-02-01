//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/26/23.
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

// third-party includes
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>

// local includes
#include "core/platform_independence/Qualifiers.h"
#include "core/linalg/Rodrigues.h"
#include "core/linalg/LinalgUtils.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::core::linalg::internal {

//TODO: move this kernel/device code to its own separate header
template<typename scalar_t>
NNRT_DEVICE_WHEN_CUDACC
inline scalar_t Sin(scalar_t value);

template<>
NNRT_DEVICE_WHEN_CUDACC
inline float Sin<float>(float value){
	return sinf(value);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline double Sin<double>(double value){
	return sin(value);
}

template<typename scalar_t>
NNRT_DEVICE_WHEN_CUDACC
inline scalar_t Cos(scalar_t value);

template<>
NNRT_DEVICE_WHEN_CUDACC
inline float Cos<float>(float value){
	return cosf(value);
}

template<>
NNRT_DEVICE_WHEN_CUDACC
inline double Cos<double>(double value){
	return cos(value);
}

template<open3d::core::Device::DeviceType TDeviceType, typename TElement>
inline void AxisAngleVectorsToMatricesRodrigues_TypeDispatched(open3d::core::Tensor& matrices, const open3d::core::Tensor& vectors) {
	o3c::Device device = vectors.GetDevice();
	auto rotation_count = vectors.GetShape(0);
	matrices = o3c::Tensor({rotation_count, 3, 3}, vectors.GetDtype(), device);
	const auto* vector_data = vectors.GetDataPtr<TElement>();
	auto* matrix_data = matrices.GetDataPtr<TElement>();
	o3c::ParallelFor(
			device, rotation_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_rotation) {
				Eigen::Map<const Eigen::Vector3<TElement>> axis_angle(vector_data + i_rotation * 3);
				TElement angle = axis_angle.norm();
				Eigen::Vector3<TElement> axis = axis_angle / angle;
				Eigen::SkewSymmetricMatrix3<TElement> axis_skew(axis);
				Eigen::Matrix3<TElement> axis_skew_dense = axis_skew.toDenseMatrix();
				if(1 == 1){
					printf("%f %f %f\n"
					       "%f %f %f\n"
					       "%f %f %f\n\n",
						   axis_skew_dense(0,0), axis_skew_dense(0,1), axis_skew_dense(0,2),
						   axis_skew_dense(1,0), axis_skew_dense(1,1), axis_skew_dense(1,2),
						   axis_skew_dense(2,0), axis_skew_dense(2,1), axis_skew_dense(2,2));
				}

				Eigen::Map<Eigen::Matrix<TElement, 3, 3, Eigen::RowMajor>> matrix(matrix_data + i_rotation * 9);
				//__DEBUG
				matrix = Eigen::Matrix<TElement, 3, 3, Eigen::RowMajor>::Zero();
				matrix.row(0) = axis_skew.toDenseMatrix();
				// matrix = Eigen::Matrix<TElement, 3, 3, Eigen::RowMajor>::Identity() + Sin(angle) * axis_skew_dense + (1-Cos(angle)) * (axis_skew_dense * axis_skew_dense);
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void AngleAxisVectorsToMatricesRodrigues(open3d::core::Tensor& matrices, const open3d::core::Tensor& vectors) {
	// tensor checks
	o3c::AssertTensorDtypes(vectors, { o3c::Float32, o3c::Float64 });
	o3c::AssertTensorShape(vectors, { utility::nullopt, 3 });
	// dispatch by data type
	DISPATCH_LINALG_DTYPE_TO_TEMPLATE(vectors.GetDtype(), [&]() {
		AxisAngleVectorsToMatricesRodrigues_TypeDispatched<TDeviceType, scalar_t>(matrices, vectors);
	});
}


} // namespace nnrt::core::linalg::internal