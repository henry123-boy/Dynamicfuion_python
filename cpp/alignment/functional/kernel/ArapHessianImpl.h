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
#include "core/linalg/BlockSparseArrowheadMatrix.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/AtomicTensor.h"
#include "core/linalg/BlockSums.h"
#include "core/linalg/DiagonalBlocks.h"
#include "core/linalg/SparseBlocks.h"
#include "core/linalg/PreconditionDiagonalBlocks.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::alignment::functional::kernel {

#define BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE 80
#define BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE 160

template<open3d::core::Device::DeviceType TDeviceType>
void ArapSparseHessianApproximation(
		core::linalg::BlockSparseArrowheadMatrix& arap_hessian_approximation,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& condensed_edge_jacobians,
		int64_t first_layer_node_count,
		int64_t second_layer_node_count,
		int64_t node_count,
		int64_t max_vertex_degree,
		float levenberg_marquardt_factor
) {
	//TODO: provisions for translation-only and rotation-only hessians (3x3 blocks)
	const int block_size = 6;
	const int block_stride = block_size * block_size;

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

	const auto first_layer_vertex_degree = std::min(max_vertex_degree, second_layer_node_count);
	const auto arrow_base_block_index = static_cast<int32_t>(first_layer_node_count); // for clarity
	const auto upper_right_wing_block_count = edge_count - first_layer_node_count * first_layer_vertex_degree;
	const auto corner_non_diagonal_block_count = edge_count - upper_right_wing_block_count;


	// input/output prep
	arap_hessian_approximation.upper_right_wing_blocks = o3c::Tensor({upper_right_wing_block_count, block_size, block_size}, o3c::Float32, device);
	auto upper_right_wing_block_data = arap_hessian_approximation.upper_right_wing_blocks.GetDataPtr<float>();
	arap_hessian_approximation.upper_right_wing_block_coordinates = o3c::Tensor({upper_right_wing_block_count, 2}, o3c::Int32, device);
	auto upper_right_wing_block_coordinate_data = arap_hessian_approximation.upper_right_wing_block_coordinates.GetDataPtr<int32_t>();


	const int64_t breadboard_width_blocks = node_count - first_layer_node_count;
	arap_hessian_approximation.upper_right_arrow_wing_breadboard = o3c::Tensor({node_count, breadboard_width_blocks}, o3c::Int16);
	arap_hessian_approximation.upper_right_arrow_wing_breadboard.Fill(-1);
	auto breadboard_data = arap_hessian_approximation.upper_right_arrow_wing_breadboard.GetDataPtr<int16_t>();

	o3c::Tensor corner_upper_blocks = o3c::Tensor({corner_non_diagonal_block_count, block_size, block_size}, o3c::Float32, device);
	auto corner_upper_block_data = corner_upper_blocks.GetDataPtr<float>();
	o3c::Tensor corner_upper_block_coordinates = o3c::Tensor({corner_non_diagonal_block_count, 2}, o3c::Int32, device);
	auto corner_upper_block_coordinate_data = corner_upper_block_coordinates.GetDataPtr<int32_t>();

	arap_hessian_approximation.arrow_base_block_index = arrow_base_block_index;

	arap_hessian_approximation.upper_column_block_lists = o3c::Tensor({breadboard_width_blocks, BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE, 2},
	                                                                  o3c::Int32, device);
	auto block_column_lookup_data = arap_hessian_approximation.upper_column_block_lists.GetDataPtr<int32_t>();
	arap_hessian_approximation.upper_row_block_lists = o3c::Tensor({node_count, BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE, 2}, o3c::Int32,
	                                                               device);
	auto block_row_lookup_data = arap_hessian_approximation.upper_row_block_lists.GetDataPtr<int32_t>();

	core::AtomicCounterArray<TDeviceType> column_block_counts(breadboard_width_blocks);
	core::AtomicCounterArray<TDeviceType> row_block_counts(node_count);


	auto edge_data = edges.GetDataPtr<int32_t>();
	auto edge_jacobian_data = condensed_edge_jacobians.GetDataPtr<float>();

	// compute upper blocks and compile breadboard + any other lookup structures for upper blocks
	o3c::ParallelFor(
			device,
			edge_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_edge) {
				Eigen::Map<const Eigen::Vector3f> dEi_dR_dense(edge_jacobian_data + i_edge * 5);
				int32_t i_node;
				int32_t j_node;

				core::kernel::Matrix3x6f dEi;
				dEi.leftCols(3) = Eigen::SkewSymmetricMatrix3<float>(dEi_dR_dense); //dEi_dR
				dEi.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 3]; //dEi_dt
				core::kernel::Matrix3x6f dEj = core::kernel::Matrix3x6f::Zero();
				dEj.rightCols(3) = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 4]; //dEj_dt

				i_node = edge_data[i_edge * 2];
				j_node = edge_data[i_edge * 2 + 1];

				if (i_node < arrow_base_block_index) {
					int64_t i_upper_right_wing_block = i_edge - corner_non_diagonal_block_count;
					Eigen::Map<core::kernel::Matrix6f> upper_block(upper_right_wing_block_data + i_upper_right_wing_block * block_stride);
					upper_block = dEi.transpose() * dEj;

					// fill breadboard
					int i_breadboard_column = j_node - arrow_base_block_index;
					breadboard_data[i_node * breadboard_width_blocks + i_breadboard_column] = static_cast<int16_t>(i_upper_right_wing_block);

					// fill block coordinate list
					upper_right_wing_block_coordinate_data[i_upper_right_wing_block * 2] = i_node;
					upper_right_wing_block_coordinate_data[i_upper_right_wing_block * 2 + 1] = j_node;

					// fill column lookup
					int i_block_in_column = column_block_counts.FetchAdd(i_breadboard_column, 1);
					if (i_block_in_column > BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE) {
						printf("Warning: encountered %i blocks for column %i, which is greater than the allowed maximum, %i. "
						       "Please either increase the BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE in code or reduce input size. "
						       "Continuing, but block-sparse matrix lookup data will be incomplete.\n", i_block_in_column, j_node,
						       BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE);
					} else {
						int i_column_lookup_item = i_breadboard_column * BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE + i_block_in_column;
						block_column_lookup_data[i_column_lookup_item * 2] = i_node;
						block_column_lookup_data[i_column_lookup_item * 2 + 1] = static_cast<int32_t>(i_edge);
					}

					// fill row lookup
					int i_block_in_row = row_block_counts.FetchAdd(i_node, 1);
					if (i_block_in_row > BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE) {
						printf("Warning: encountered %i blocks for row %i, which is greater than the allowed maximum, %i. "
						       "Please either increase the BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE in code or reduce input size. "
						       "Continuing, but block-sparse matrix lookup data will be incomplete.\n", i_block_in_row, i_node,
						       BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE);
					} else {
						int i_row_lookup_item = i_node * BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE + i_block_in_row;
						block_row_lookup_data[i_row_lookup_item * 2] = j_node;
						block_row_lookup_data[i_row_lookup_item * 2 + 1] = static_cast<int32_t>(i_edge);
					}
				} else {
					Eigen::Map<core::kernel::Matrix6f> upper_block(corner_upper_block_data + i_edge * block_stride);
					upper_block = dEi.transpose() * dEj;

					// fill block coordinate list
					corner_upper_block_coordinate_data[i_edge * 2] = i_node;
					corner_upper_block_coordinate_data[i_edge * 2 + 1] = j_node;
				}
			}
	);

	arap_hessian_approximation.upper_column_block_counts = column_block_counts.AsTensor(true);
	arap_hessian_approximation.upper_row_block_counts = row_block_counts.AsTensor(true);

#ifndef __CUDACC__
	std::vector<std::atomic<float>> hessian_blocks_diagonal_atomic(node_count * 36);
#endif
	o3c::Tensor diagonal_components = o3c::Tensor({edge_count * 2, block_size, block_size}, o3c::Float32, device);
	auto diagonal_component_data = diagonal_components.GetDataPtr<float>();

	o3c::Tensor diagonal_component_block_indices = o3c::Tensor({edge_count * 2, block_size, block_size}, o3c::Int32, device);
	auto diagonal_component_block_index_data = diagonal_component_block_indices.GetDataPtr<int>();
	// compute diagonal block components
	o3c::ParallelFor(
			device,
			edge_count * 2,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t i_edge = workload_idx / 2;

				int64_t i_node_in_edge = workload_idx % 2;
				int64_t node_index = edge_data[workload_idx];

				Eigen::Map<core::kernel::Matrix6f> h_block_update(diagonal_component_data + workload_idx * block_stride);
				diagonal_component_block_index_data[workload_idx] = static_cast<int>(node_index);
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
			}
	);

	core::AtomicTensor<TDeviceType, float> diagonal_blocks_atomic({node_count, block_size, block_size}, device);
	core::linalg::internal::ComputeBlockSums(diagonal_blocks_atomic, node_count, diagonal_components, diagonal_component_block_indices, edge_count * 2);
	auto diagonal_blocks = diagonal_blocks_atomic.AsTensor(false);

	if(levenberg_marquardt_factor > 0.f){
		core::linalg::internal::PreconditionDiagonalBlocks<TDeviceType>(diagonal_blocks, levenberg_marquardt_factor);
	}

	arap_hessian_approximation.stem_diagonal_blocks = diagonal_blocks.Slice(0, 0, arrow_base_block_index).Clone();


	arap_hessian_approximation.corner_dense_matrix =
			o3c::Tensor({breadboard_width_blocks * block_size, breadboard_width_blocks * block_size}, o3c::Float32, device);

	o3c::Tensor corner_diagonal_blocks = diagonal_blocks.Slice(0, arrow_base_block_index, node_count);

	core::linalg::internal::FillInDiagonalBlocks<TDeviceType>(arap_hessian_approximation.corner_dense_matrix, corner_diagonal_blocks);
	core::linalg::internal::FillInSparseBlocks<TDeviceType>(arap_hessian_approximation.corner_dense_matrix, corner_upper_blocks,
	                                                        corner_upper_block_coordinates, false);

	// TODO: potentially, optimize-out -- this should be unnecessary for computation

#ifndef __CUDACC__
	// copy over data from atomics to tensor on CPU
	o3c::ParallelFor(
			device,
			node_count * 36,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				diagonal_component_data[workload_idx] = hessian_blocks_diagonal_atomic[workload_idx].load();
			}
	);
#endif

}


} // namespace nnrt::alignment::functional::kernel