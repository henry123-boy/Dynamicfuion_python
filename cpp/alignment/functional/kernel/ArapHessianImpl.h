//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/2/23.
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

// local includes
#include "alignment/functional/kernel/ArapHessian.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/functional/ParallelPrefixScan.h"
#include "core/platform_independence/Atomics.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::alignment::functional::kernel {


template<open3d::core::Device::DeviceType TDeviceType>
void ArapSparseHessianApproximation(
		open3d::core::Tensor& arap_hessian_blocks_upper,
		open3d::core::Tensor& arap_hessian_upper_block_coordinates,
		open3d::core::Tensor& arap_hessian_upper_block_breadboard,
		open3d::core::Tensor& arap_hessian_blocks_diagonal,

		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& condensed_edge_jacobians,
		int64_t first_layer_node_count,
		int64_t node_count
) {
	// checks & counts
	o3c::Device device = edges.GetDevice();
	int64_t edge_count = edges.GetLength();

	o3c::AssertTensorShape(edges, { edge_count, 2 });
	o3c::AssertTensorDtype(edges, o3c::Int32);

	o3c::AssertTensorShape(condensed_edge_jacobians, { edge_count, 5 });
	o3c::AssertTensorDtype(condensed_edge_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(condensed_edge_jacobians, device);

	if (first_layer_node_count < 1 || node_count < 1) {
		utility::LogError("first_layer_node_count (currently, {}) and node_count (currently, {}) both need to be positive integers.",
		                  first_layer_node_count, node_count);
	}
	if (first_layer_node_count > node_count) {
		utility::LogError("first_layer_node_count (currently, {}) cannot exceed node_count (currently, {}).", first_layer_node_count, node_count);
	}

	if (node_count > 32767) {
		utility::LogError("node_count (currently, {}) cannot exceed int16_t max, which is 32767.", node_count);
	}

	// input/output prep
	arap_hessian_blocks_upper = o3c::Tensor({edge_count, 6, 6}, o3c::Float32, device);
	auto upper_block_data = arap_hessian_blocks_upper.GetDataPtr<float>();
	arap_hessian_upper_block_coordinates = o3c::Tensor({edge_count, 2}, o3c::Int32, device);
	auto upper_block_coordinate_data = arap_hessian_upper_block_coordinates.GetDataPtr<int32_t>();
	arap_hessian_blocks_diagonal = o3c::Tensor({node_count, 6, 6}, o3c::Float32, device);
	auto diagonal_block_data = arap_hessian_blocks_diagonal.GetDataPtr<float>();
	int64_t breadboard_width = node_count - first_layer_node_count;
	arap_hessian_upper_block_breadboard = o3c::Tensor({node_count, breadboard_width}, o3c::Int16);
	arap_hessian_upper_block_breadboard.Fill(-1);
	auto breadboard_data = arap_hessian_upper_block_breadboard.GetDataPtr<int16_t>();

	auto edge_data = edges.GetDataPtr<int32_t>();
	auto edge_jacobian_data = condensed_edge_jacobians.GetDataPtr<float>();

	// compute upper blocks and compile breadboard
	o3c::ParallelFor(
			device,
			edge_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_edge) {
				Eigen::Map<const Eigen::Vector3f> dEi_dR_dense(edge_jacobian_data + i_edge * 5);

				core::kernel::Matrix3x6f dEi;
				dEi.leftCols(3) = Eigen::SkewSymmetricMatrix3<float>(dEi_dR_dense); //dEi_dR
				dEi.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 3]; //dEi_dt
				core::kernel::Matrix3x6f dEj = core::kernel::Matrix3x6f::Zero();
				dEj.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 4]; //dEj_dt

				Eigen::Map<core::kernel::Matrix6f> upper_block(upper_block_data + i_edge * 36);
				upper_block = dEi.transpose() * dEj;

				int32_t i_node = edge_data[i_edge * 2];
				int32_t j_node = edge_data[i_edge * 2 + 1];
				breadboard_data[i_node * breadboard_width + (j_node - first_layer_node_count)] = static_cast<int16_t>(i_edge);

				upper_block_coordinate_data[i_edge * 2] = i_node;
				upper_block_coordinate_data[i_edge * 2 + 1] = j_node;
			}
	);

#ifndef __CUDACC__
	std::vector<std::atomic<float>> hessian_blocks_diagonal_atomic(node_count * 36);
#endif
	// compute diagonal blocks
	o3c::ParallelFor(
			device,
			edge_count * 2,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_edge = workload_idx / 2;
				int64_t i_node_in_edge = workload_idx % 2;
				int64_t node_index = edge_data[workload_idx];

				core::kernel::Matrix6f h_block_update;
				if (i_node_in_edge == 0) { // "i"
					core::kernel::Matrix3x6f dEi;
					Eigen::Map<const Eigen::Vector3f> dEi_dR_dense(edge_jacobian_data + i_edge * 5);
					dEi.leftCols(3) = Eigen::SkewSymmetricMatrix3<float>(dEi_dR_dense); //dEi_dR
					dEi.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 3]; //dEi_dt
					h_block_update = dEi.transpose() * dEi;
				} else { // "j"
					core::kernel::Matrix3x6f dEj = core::kernel::Matrix3x6f::Zero();
					dEj.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 4]; //dEj_dt
					h_block_update = dEj.transpose() * dEj;
				}
#ifdef __CUDACC__
				auto current_diagonal_block_data = diagonal_block_data + node_index * 36;
#endif
				for(int i_coefficient = 0; i_coefficient < 36; i_coefficient++){
#ifdef __CUDACC__
					atomicAdd(current_diagonal_block_data + i_coefficient, h_block_update.coeff(i_coefficient));
#else
					atomicAdd_CPU(hessian_blocks_diagonal_atomic[node_index * 36 + i_coefficient], h_block_update.coeff(i_coefficient));
#endif
				}

			}
	);


#ifndef __CUDACC__
	// copy over data from atomics to tensor on CPU
	o3c::ParallelFor(
			device,
			node_count * 36,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				diagonal_block_data[workload_idx] = hessian_blocks_diagonal_atomic[workload_idx].load();
			}
	);
#endif

}


} // namespace nnrt::alignment::functional::kernel