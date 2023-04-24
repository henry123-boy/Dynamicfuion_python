//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/24/23.
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
#include <Eigen/Dense>

// local includes
#include "geometry/functional/kernel/ComputeDistanceMatrix.h"
#include "core/platform_independence/Qualifiers.h"
namespace o3c = open3d::core;

namespace nnrt::geometry::functional::kernel{

template <open3d::core::Device::DeviceType TDeviceType>
void ComputeDistanceMatrix(open3d::core::Tensor& distance_matrix, const open3d::core::Tensor& point_set1, const open3d::core::Tensor& point_set2){
	auto device = point_set1.GetDevice();
	int64_t point_count1 = point_set1.GetShape(0);
	int64_t point_count2 = point_set2.GetShape(0);
	o3c::AssertTensorShape(point_set1, {point_count1, 3});
	o3c::AssertTensorDtype(point_set1, o3c::Float32);
	o3c::AssertTensorDevice(point_set2, device);
	o3c::AssertTensorShape(point_set2, {point_count2, 3});
	o3c::AssertTensorDtype(point_set2, o3c::Float32);

	distance_matrix = o3c::Tensor({point_count1, point_count2}, o3c::Float32, device);

	auto point_set1_data = point_set1.GetDataPtr<float>();
	auto point_set2_data = point_set2.GetDataPtr<float>();
	auto distance_data = distance_matrix.GetDataPtr<float>();

	o3c::ParallelFor(
		device,
		point_count1 * point_count2,
		NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_matrix_entry){
			int64_t i_set1_point = i_matrix_entry / point_count2;
			int64_t i_set2_point = i_matrix_entry % point_count2;
			Eigen::Map<const Eigen::Vector3f> set1_point(point_set1_data + i_set1_point * 3);
			Eigen::Map<const Eigen::Vector3f> set2_point(point_set2_data + i_set2_point * 3);
			distance_data[i_matrix_entry] = (set1_point - set2_point).norm();
		}
	);
}

} // namespace nnrt::geometry::functional::kernel