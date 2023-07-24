//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/10/23.
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
#include "alignment/kernel/DeformableMeshToImageFitter.h"
#include "core/platform_independence/Qualifiers.h"
#include "core/kernel/MathTypedefs.h"
#include "core/linalg/KroneckerTensorProduct.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/AtomicTensor.h"
#include "core/platform_independence/Atomics.h"
#include "geometry/functional/kernel/Defines.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;

#define MAX_PIXELS_PER_NODE 4000


namespace nnrt::alignment::kernel {

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Swap(TScalar* a, TScalar* b) {
	TScalar tmp = *a;
	*a = *b;
	*b = tmp;
}

template<typename TScalar>
inline NNRT_DEVICE_WHEN_CUDACC void Heapify(TScalar* array, int length, int root) {
	int largest = root;
	int l = 2 * root + 1;
	int r = 2 * root + 2;

	if (l < length && array[l] > array[largest]) {
		largest = l;
	}
	if (r < length && array[r] > array[largest]) {
		largest = r;
	}
	if (largest != root) {
		Swap<TScalar>(&array[root], &array[largest]);
		Heapify<TScalar>(array, length, largest);
	}
}

template<typename TScalar>
NNRT_DEVICE_WHEN_CUDACC void HeapSort(TScalar* array, int length) {
	for (int i = length / 2 - 1; i >= 0; i--) Heapify(array, length, i);

	for (int i = length - 1; i > 0; i--) {
		Swap<TScalar>(&array[0], &array[i]);
		Heapify<TScalar>(array, i, 0);
	}
}

template<open3d::core::Device::DeviceType TDevice>
void ConvertPixelVertexAnchorJacobiansToNodeJacobians(
		open3d::core::Tensor& node_jacobians,
		open3d::core::Tensor& node_jacobian_ranges,
		open3d::core::Tensor& node_pixel_indices,
		open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_counts,
		const open3d::core::Tensor& pixel_jacobians
) {
	// === dimension, type, and device tensor checks ===
	int64_t node_count = node_pixel_counts.GetShape(0);
	o3c::Device device = node_pixel_counts.GetDevice();
	o3c::AssertTensorShape(node_pixel_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_counts, o3c::Int32);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);

	o3c::AssertTensorShape(pixel_jacobians, { utility::nullopt, utility::nullopt, 6 });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	// === get access to input arrays ===
	auto node_pixel_index_jagged_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_counts.GetDataPtr<int32_t>();
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();

	// === set up atomic counter ===
	NNRT_DECLARE_ATOMIC(uint32_t, total_jacobian_count);
	NNRT_INITIALIZE_ATOMIC(uint32_t, total_jacobian_count, 0L);

	// === set up output tensor to store ranges ===
	node_jacobian_ranges = o3c::Tensor({node_count, 2}, o3c::Int64, device);
	auto node_jacobian_range_data = node_jacobian_ranges.GetDataPtr<int64_t>();
	// === loop over all nodes and sort all entries by jacobian address
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				int* node_jacobian_list_start =
						node_pixel_index_jagged_data + (node_index * MAX_PIXELS_PER_NODE);
				int node_jacobian_list_length = node_pixel_count_data[node_index];
				// sort the anchor jacobian addresses
				HeapSort(node_jacobian_list_start, node_jacobian_list_length);

				node_jacobian_range_data[node_index * 2 + 1] = node_jacobian_list_length;
				NNRT_ATOMIC_ADD(total_jacobian_count, static_cast<uint32_t>(node_jacobian_list_length));
			}
	);

	node_jacobians = o3c::Tensor({NNRT_GET_ATOMIC_VALUE_HOST(total_jacobian_count), 6L}, o3c::Float32,
	                             device);

	NNRT_CLEAN_UP_ATOMIC(total_jacobian_count);
	auto node_jacobian_data = node_jacobians.GetDataPtr<float>();

	node_pixel_indices = o3c::Tensor({node_jacobians.GetShape(0)}, o3c::Int32, device);
	auto node_pixel_index_compact_data = node_pixel_indices.GetDataPtr<int32_t>();

	// === loop over all nodes again, this time aggregating their jacobians for each pixel they affect
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				// figure out where to start filling in the jacobians for the current node
				int node_jacobian_start_index = 0;
				for (int i_node = 0; i_node < node_index - 1; i_node++) {
					node_jacobian_start_index += node_jacobian_range_data[i_node * 2 + 1];
				}
				node_jacobian_range_data[node_index * 2] = node_jacobian_start_index;

				// source data to get anchor jacobians from
				int* node_pixel_index_jagged_start = node_pixel_index_jagged_data + (node_index * MAX_PIXELS_PER_NODE);
				int node_pixel_count = node_pixel_count_data[node_index];

				// loop over source data and fill in node jacobians & corresponding pixel indices
				int i_node_jacobian = 0;
				for (int i_node_pixel = 0; i_node_pixel < node_pixel_count; i_node_pixel++) {
					int pixel_jacobian_address = node_pixel_index_jagged_start[i_node_pixel];
					Eigen::Map<const Eigen::RowVector<float, 6>>
							pixel_jacobian(pixel_jacobian_data + pixel_jacobian_address);
					Eigen::Map<Eigen::RowVector<float, 6>>
							node_jacobian(node_jacobian_data + node_jacobian_start_index + i_node_jacobian * 6);
					node_pixel_index_compact_data[node_jacobian_start_index + i_node_jacobian]
							= pixel_jacobian_address / 6;
					node_jacobian = pixel_jacobian;
				}
			}
	);
}

// @formatter:off
NNRT_CONSTANT_WHEN_CUDACC const int three_column_0_lookup_table[6] = {
			0, 0, 0,
			   1, 1,
			      2
};
NNRT_CONSTANT_WHEN_CUDACC const int three_column_1_lookup_table[6] = {
		0, 1, 2,
		   1, 2,
		      2
};
NNRT_CONSTANT_WHEN_CUDACC const int six_column_0_lookup_table[21] = {
            0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
                  2, 2, 2, 2,
                     3, 3, 3,
                        4, 4,
                           5
};
NNRT_CONSTANT_WHEN_CUDACC const int six_column_1_lookup_table[21] = {
            0, 1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
                  2, 3, 4, 5,
                     3, 4, 5,
                        4, 5,
                           5
};
// @formatter:on



//TODO: can optimize: don't fill in lower triangle at all, use a batched triangular solver instead of Cholesky?


template<open3d::core::Device::DeviceType TDevice, IterationMode TIterationMode>
void ComputeHessianApproximationBlocks_UnorderedNodePixels_Generic(
		open3d::core::Tensor& hessian_approximation_blocks,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts
) {
	// === dimension, type, and device tensor checks ===
	int64_t node_count = node_pixel_jacobian_counts.GetShape(0);
	o3c::Device device = node_pixel_jacobian_counts.GetDevice();

	o3c::AssertTensorShape(node_pixel_jacobian_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_jacobian_counts, o3c::Int32);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);

	int jacobian_stride;
	int64_t unique_entry_count_per_block;
	if (TIterationMode == IterationMode::ALL) {
		jacobian_stride = 6;
		unique_entry_count_per_block = 21;
	} else {
		jacobian_stride = 3;
		unique_entry_count_per_block = 6;
	}

	const int jacobian_block_stride = jacobian_stride * jacobian_stride;

	o3c::AssertTensorShape(pixel_jacobians, { utility::nullopt, utility::nullopt, jacobian_stride });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	// === get access to input arrays ===
	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_jacobian_counts.GetDataPtr<int32_t>();
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();

	/*
	 * For "ALL" iteration:
	 * Each node has 6 delta components (3 rotational and 3 translational).
	 * These would be represented in a dense jacobian as rows within the same 6 columns, with each row
	 * corresponding to a separate pixel.
	 * Block-diagonal optimization assumes that nodes have no influence on each
	 * other, so we can imagine separate J_node matrices for each node, of size [pixel count x 6]
	 * Each block can then be found via (J_node^T)*J_node, and will have size 6x6.
	 * Each entry in a single 6x6 block is a dot product of one of the six columns with another, hence:
	 *    1. The number of unique column combinations is (2 + 6 - 1)! / (2!*(6-1)!), or 21,
	 *       so there are only 21 unique entries per block
	 *    2. Due to commutativity of addition, it doesn't matter in what order the addends of each dot product are
	 *       retrieved from memory, as long as we always sample their factors from the same row
	 *
	 * For "TRANSLATION_ONLY" or "ROTATION_ONLY" iterations:
	 * we follow exactly the same logic as above, except
	 * now we have only 3 delta components, and hence 6 unique elements per each 3x3 block.
	 */

	// initialize output structure
	hessian_approximation_blocks = o3c::Tensor({node_count, jacobian_stride, jacobian_stride}, o3c::Float32, device);
	auto hessian_approximation_block_data = hessian_approximation_blocks.GetDataPtr<float>();

	o3c::ParallelFor(
			device, unique_entry_count_per_block * node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_index) {
				int i_node = static_cast<int>(workload_index / unique_entry_count_per_block);
				int i_unique_element_in_block = static_cast<int>(workload_index % unique_entry_count_per_block);
				int i_column_0;
				int i_column_1;

				if (TIterationMode == IterationMode::ALL) {
					i_column_0 = six_column_0_lookup_table[i_unique_element_in_block];
					i_column_1 = six_column_1_lookup_table[i_unique_element_in_block];
				} else {
					i_column_0 = three_column_0_lookup_table[i_unique_element_in_block];
					i_column_1 = three_column_1_lookup_table[i_unique_element_in_block];
				}

				int node_pixel_jacobian_list_length = node_pixel_count_data[i_node];
				float column_product = 0.0;
				for (int i_node_pixel_jacobian = 0; i_node_pixel_jacobian < node_pixel_jacobian_list_length; i_node_pixel_jacobian++) {
					int pixel_node_jacobian_address = node_pixel_jacobian_index_data[i_node * MAX_PIXELS_PER_NODE + i_node_pixel_jacobian];
					const float* jacobian = pixel_jacobian_data + pixel_node_jacobian_address;
					float addend = jacobian[i_column_0] * jacobian[i_column_1];
					column_product += addend;
				}
				// parenthesis for clarity
				hessian_approximation_block_data[(i_node * jacobian_block_stride) + (i_column_0 * jacobian_stride) + i_column_1] = column_product;
				hessian_approximation_block_data[(i_node * jacobian_block_stride) + (i_column_1 * jacobian_stride) + i_column_0] = column_product;
			}
	);
}

template<open3d::core::Device::DeviceType TDevice>
void ComputeDepthHessianApproximationBlocks_UnorderedNodePixels(
		open3d::core::Tensor& hessian_approximation_blocks,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		IterationMode mode
) {
	switch (mode) {
		case ALL:
			ComputeHessianApproximationBlocks_UnorderedNodePixels_Generic<TDevice, IterationMode::ALL>(
					hessian_approximation_blocks, pixel_jacobians, node_pixel_jacobian_indices, node_pixel_jacobian_counts
			);
			break;
		case TRANSLATION_ONLY:
			ComputeHessianApproximationBlocks_UnorderedNodePixels_Generic<TDevice, IterationMode::TRANSLATION_ONLY>(
					hessian_approximation_blocks, pixel_jacobians, node_pixel_jacobian_indices, node_pixel_jacobian_counts
			);
			break;
		case ROTATION_ONLY:
			ComputeHessianApproximationBlocks_UnorderedNodePixels_Generic<TDevice, IterationMode::ROTATION_ONLY>(
					hessian_approximation_blocks, pixel_jacobians, node_pixel_jacobian_indices, node_pixel_jacobian_counts
			);
			break;
	}
}

namespace internal {
template<IterationMode TIterationMode>
struct Gradient;

template<>
struct Gradient<IterationMode::ALL> {
	typedef typename Eigen::Vector<float, 6> VectorType;
};

template<>
struct Gradient<IterationMode::ROTATION_ONLY> {
	typedef typename Eigen::Vector<float, 3> VectorType;
};

template<>
struct Gradient<IterationMode::TRANSLATION_ONLY> {
	typedef typename Eigen::Vector<float, 3> VectorType;
};

} // internal

template<open3d::core::Device::DeviceType TDevice, IterationMode TIterationMode>
void ComputeNegativeGradient_UnorderedNodePixels_Generic(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex
) {
	// === dimension, type, and device tensor checks ===
	o3c::Device device = pixel_jacobians.GetDevice();
	int64_t pixel_count = pixel_jacobians.GetShape(0);
	int64_t max_jacobian_count = pixel_jacobians.GetShape(1);

	int jacobian_stride;
	if (TIterationMode == IterationMode::ALL) {
		jacobian_stride = 6;
	} else {
		jacobian_stride = 3;
	}

	o3c::AssertTensorShape(pixel_jacobians, { pixel_count, max_jacobian_count, jacobian_stride });
	o3c::AssertTensorDtype(pixel_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(pixel_jacobians, device);

	o3c::AssertTensorShape(residuals, { pixel_count });
	o3c::AssertTensorDtype(residuals, o3c::Float32);
	o3c::AssertTensorDevice(residuals, device);

	o3c::AssertTensorShape(residual_mask, { pixel_count });
	o3c::AssertTensorDtype(residual_mask, o3c::Bool);
	o3c::AssertTensorDevice(residual_mask, device);

	int64_t node_count = node_pixel_jacobian_counts.GetShape(0);
	o3c::AssertTensorShape(node_pixel_jacobian_counts, { node_count });
	o3c::AssertTensorDtype(node_pixel_jacobian_counts, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_counts, device);

	o3c::AssertTensorShape(node_pixel_jacobian_indices, { node_count, MAX_PIXELS_PER_NODE });
	o3c::AssertTensorDtype(node_pixel_jacobian_indices, o3c::Int32);
	o3c::AssertTensorDevice(node_pixel_jacobian_indices, device);


	// === get access to input arrays ===
	auto pixel_jacobian_data = pixel_jacobians.GetDataPtr<float>();
	auto residual_data = residuals.GetDataPtr<float>();
	auto residual_mask_data = static_cast<const unsigned char*>(residual_mask.GetDataPtr());
	auto node_pixel_jacobian_index_data = node_pixel_jacobian_indices.GetDataPtr<int32_t>();
	auto node_pixel_count_data = node_pixel_jacobian_counts.GetDataPtr<int32_t>();

	// === initialize output structures ===
	negative_gradient = o3c::Tensor::Zeros({node_count * jacobian_stride}, o3c::Float32, device);
	auto negative_gradient_data = negative_gradient.GetDataPtr<float>();

	//TODO: this is extremely inefficient. There must be some parallel reduction we can apply here, but everything is difficult.
	// The current code does a variable number of essentially random (global) memory accesses for each node (as many as there are jacobians,
	// i.e. pixels controlled by each node.  These are ordered, potentially not at all cache-able. In addition, it makes as many random accesses
	// to the residual array in order to perform the actual multiplication. These results are summed together to produce entries of the gradient.
	o3c::ParallelFor(
			device, node_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t node_index) {
				int node_pixel_jacobian_list_length = node_pixel_count_data[node_index];
				typename internal::Gradient<TIterationMode>::VectorType node_pixel_gradient = internal::Gradient<TIterationMode>::VectorType::Zero();
				// if (TIterationMode == IterationMode::ALL) {
				// 	node_pixel_gradient << 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
				// } else {
				// 	node_pixel_gradient << 0.f, 0.f, 0.f;
				// }
				for (int i_node_pixel_jacobian = 0; i_node_pixel_jacobian < node_pixel_jacobian_list_length; i_node_pixel_jacobian++) {
					int pixel_node_jacobian_address = node_pixel_jacobian_index_data[node_index * MAX_PIXELS_PER_NODE + i_node_pixel_jacobian];
					int i_pixel = pixel_node_jacobian_address / (max_anchor_count_per_vertex * 3 * jacobian_stride);
					if (!residual_mask_data[i_pixel]) continue;
					Eigen::Map<const typename internal::Gradient<TIterationMode>::VectorType> node_pixel_jacobian(
							pixel_jacobian_data + pixel_node_jacobian_address);
					float residual = residual_data[i_pixel];
					node_pixel_gradient += node_pixel_jacobian * residual;
				}
				Eigen::Map<typename internal::Gradient<TIterationMode>::VectorType> negative_node_gradient(
						negative_gradient_data + node_index * jacobian_stride);
				negative_node_gradient -= node_pixel_gradient;
			}
	);
}

template<open3d::core::Device::DeviceType TDevice>
void ComputeNegativeDepthGradient_UnorderedNodePixels(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& residual_mask,
		const open3d::core::Tensor& pixel_jacobians,
		const open3d::core::Tensor& node_pixel_jacobian_indices,
		const open3d::core::Tensor& node_pixel_jacobian_counts,
		int max_anchor_count_per_vertex,
		IterationMode mode
) {
	switch (mode) {
		case ALL:
			ComputeNegativeGradient_UnorderedNodePixels_Generic<TDevice, IterationMode::ALL>(
					negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices,
					node_pixel_jacobian_counts, max_anchor_count_per_vertex
			);
			break;
		case TRANSLATION_ONLY:
			ComputeNegativeGradient_UnorderedNodePixels_Generic<TDevice, IterationMode::TRANSLATION_ONLY>(
					negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices,
					node_pixel_jacobian_counts, max_anchor_count_per_vertex
			);
			break;
		case ROTATION_ONLY:
			ComputeNegativeGradient_UnorderedNodePixels_Generic<TDevice, IterationMode::ROTATION_ONLY>(
					negative_gradient, residuals, residual_mask, pixel_jacobians, node_pixel_jacobian_indices,
					node_pixel_jacobian_counts, max_anchor_count_per_vertex
			);
			break;
	}
}

enum class EdgeJacobianPortion : int64_t {
	NODE_I_ROTATION = 0,
	NODE_I_TRANSLATION = 1,
	NODE_J_TRANSLATION = 2
};

template<open3d::core::Device::DeviceType TDevice, IterationMode TIterationMode>
void ComputeNegativeArapGradient_ModeDispatched(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& edge_jacobians,
		const open3d::core::Tensor& edges,
		int64_t node_count
) {
	// counts & checks
	o3c::Device device = edges.GetDevice();

	int64_t edge_count = edges.GetShape(0);
	o3c::AssertTensorShape(edges, { edge_count, 2 });
	o3c::AssertTensorDtype(edges, o3c::Int32);

	o3c::AssertTensorShape(edge_jacobians, { edge_count, 5 });
	o3c::AssertTensorDtype(edge_jacobians, o3c::Float32);
	o3c::AssertTensorDevice(edge_jacobians, device);

	o3c::AssertTensorShape(residuals, { edge_count * 3 });
	o3c::AssertTensorDtype(residuals, o3c::Float32);
	o3c::AssertTensorDevice(residuals, device);

	int64_t jacobian_stride;
	if (TIterationMode == IterationMode::ALL) {
		jacobian_stride = 6;
	} else {
		jacobian_stride = 3;
	}

	// prep outputs & data access
	core::AtomicTensor<TDevice, float> negative_gradient_atomic({node_count * jacobian_stride}, device);

	auto edge_data = edges.GetDataPtr<int32_t>();
	auto edge_jacobian_data = edge_jacobians.GetDataPtr<float>();
	auto residual_data = residuals.GetDataPtr<float>();

	switch (TIterationMode) {
		case ALL: {
			o3c::ParallelFor(
					device, edge_count * 3,
					NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_workload) {
						int64_t i_edge = i_workload / 3;
						auto jacobian_portion = static_cast<EdgeJacobianPortion>(i_workload % 3);
						Eigen::Map<const Eigen::Vector3f> dEi_dR_dense(edge_jacobian_data + i_edge * 5);
						Eigen::Map<const Eigen::Vector3f> edge_residual(residual_data + i_edge * 3);
						core::kernel::Matrix3f dE;
						int32_t node_index = 0;
						int32_t offset_in_gradient = 0;
						switch(jacobian_portion){
							case EdgeJacobianPortion::NODE_I_ROTATION:
								// TODO: (potential optimization) this still more operations than i_translation and j_translation cases require
								//  (which can more efficiently be done using scalar-vector multiplication instead of matrix-vector).
								//  The smarter way for CUDA would be to use intermediate value arrays for the i_rotation (and, maybe, i_translation)
								//  cases instead -- there are consistent non-zero "diagonal" 6x12 blocks in J^(T), [de_Ri de_ti].
								//  Think about how to more efficiently do skew-symmetric.T x 3-vector multiplications on CUDA.
								//  Then, the blocks can be added up through some form of reduction. Likewise can be done for j_translation, but schema
								//  for bottom (denser) part of J.T is different (look at matrix sparsity diagram)
								dE = Eigen::SkewSymmetricMatrix3<float>(dEi_dR_dense).transpose();
								node_index = edge_data[i_edge * 2];
								offset_in_gradient = 0;
								break;
							case EdgeJacobianPortion::NODE_I_TRANSLATION:
								// not the most efficient way, but, for now, we care more about thread divergence here (for CUDA version).
								dE = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 3];
								node_index = edge_data[i_edge * 2];
								offset_in_gradient = 3;
								break;
							case EdgeJacobianPortion::NODE_J_TRANSLATION:
								// not the most efficient way, but, for now, we care more about thread divergence here (for CUDA version).
								dE = core::kernel::Matrix3f::Identity() * edge_jacobian_data[i_edge * 5 + 4];
								node_index = edge_data[i_edge * 2 + 1];
								offset_in_gradient = 3;
								break;
						}
						Eigen::Vector3f negative_gradient_addend = dE * edge_residual;
						int64_t first_element_offset = node_index * jacobian_stride + offset_in_gradient;
#ifdef __CUDACC__
						#pragma unroll
#endif
						for (int i_element = 0; i_element < 3; i_element++){
							// subtract (instead of add) to make negative gradient
							negative_gradient_atomic.FetchSub(first_element_offset + i_element, negative_gradient_addend.coeff(i_element));
						}
					}
			);
		}
			break;
		case ROTATION_ONLY:
			o3c::ParallelFor(
					device, edge_count,
					NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_edge) {
						Eigen::Map<const Eigen::Vector3f> dEi_dR_dense(edge_jacobian_data + i_edge * 5);
						Eigen::Map<const Eigen::Vector3f> edge_residual(residual_data + i_edge * 3);
						core::kernel::Matrix3f dE;
						int32_t node_index = edge_data[i_edge * 2];
						Eigen::Vector3f negative_gradient_addend = Eigen::SkewSymmetricMatrix3<float>(dEi_dR_dense).transpose() * edge_residual;
						int64_t first_element_offset = node_index * jacobian_stride;
#ifdef __CUDACC__
#pragma unroll
#endif
						for (int i_element = 0; i_element < 3; i_element++){
							// subtract (instead of add) to make negative gradient
							negative_gradient_atomic.FetchSub(first_element_offset + i_element, negative_gradient_addend.coeff(i_element));
						}
 					}
			);
			break;
		case TRANSLATION_ONLY:
			o3c::ParallelFor(
					device, edge_count*2,
					NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_workload) {
						int64_t i_edge = i_workload / 2;
						auto jacobian_portion = static_cast<EdgeJacobianPortion>(1 + i_workload % 2);
						Eigen::Map<const Eigen::Vector3f> edge_residual(residual_data + i_edge * 3);
						float factor = 0.f;
						int32_t node_index = 0;
						int32_t offset_in_gradient = 0;
						switch(jacobian_portion){
							case EdgeJacobianPortion::NODE_I_TRANSLATION:
								// not the most efficient way, but, for now, we care more about thread divergence here (for CUDA version).
								factor = edge_jacobian_data[i_edge * 5 + 3];
								node_index = edge_data[i_edge * 2];
								offset_in_gradient = 0;
								break;
							case EdgeJacobianPortion::NODE_J_TRANSLATION:
								// not the most efficient way, but, for now, we care more about thread divergence here (for CUDA version).
								factor = edge_jacobian_data[i_edge * 5 + 4];
								node_index = edge_data[i_edge * 2 + 1];
								offset_in_gradient = 3;
								break;
							default:
								printf("Warning: logic error in ComputeNegativeArapGradient_ModeDispatched\n");
								break;
						}
						Eigen::Vector3f negative_gradient_addend = factor * edge_residual;
						int64_t first_element_offset = node_index * jacobian_stride + offset_in_gradient;
#ifdef __CUDACC__
#pragma unroll
#endif
						for (int i_element = 0; i_element < 3; i_element++){
							// subtract (instead of add) to make negative gradient
							negative_gradient_atomic.FetchSub(first_element_offset + i_element, negative_gradient_addend.coeff(i_element));
						}
					}
			);
			break;
	}
	negative_gradient = negative_gradient_atomic.AsTensor(true);

}

template<open3d::core::Device::DeviceType TDevice>
void ComputeNegativeArapGradient(
		open3d::core::Tensor& negative_gradient,
		const open3d::core::Tensor& residuals,
		const open3d::core::Tensor& edge_jacobians,
		const open3d::core::Tensor& edges,
		int64_t node_count,
		IterationMode mode
) {
	switch (mode) {
		case ALL:
			ComputeNegativeArapGradient_ModeDispatched<TDevice, IterationMode::ALL>(
					negative_gradient, residuals, edge_jacobians, edges, node_count
			);
			break;
		case TRANSLATION_ONLY:
			ComputeNegativeArapGradient_ModeDispatched<TDevice, IterationMode::TRANSLATION_ONLY>(
					negative_gradient, residuals, edge_jacobians, edges, node_count
			);
			break;
		case ROTATION_ONLY:
			ComputeNegativeArapGradient_ModeDispatched<TDevice, IterationMode::ROTATION_ONLY>(
					negative_gradient, residuals, edge_jacobians, edges, node_count
			);
			break;
	}
}

template<open3d::core::Device::DeviceType TDevice>
void ComputeArapResiduals_FixedCoverageWeight(
		open3d::core::Tensor& edge_residuals,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& edge_layer_indices,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& node_rotations,
		const open3d::core::Tensor& layer_decimation_radii,
		float regularization_weight
) {
	// counts & checks
	int64_t edge_count = edges.GetLength();
	int64_t layer_count = layer_decimation_radii.GetLength();
	int64_t node_count = node_positions.GetLength();

	o3c::Device device = edges.GetDevice();

	o3c::AssertTensorShape(edges, { edge_count, 2 });
	o3c::AssertTensorDtype(edges, o3c::Int32);

	o3c::AssertTensorShape(edge_layer_indices, { edge_count });
	o3c::AssertTensorDtype(edge_layer_indices, o3c::Int8);
	o3c::AssertTensorDevice(edge_layer_indices, device);

	o3c::AssertTensorShape(node_positions, { node_count, 3 });
	o3c::AssertTensorDtype(node_positions, o3c::Float32);
	o3c::AssertTensorDevice(node_positions, device);

	o3c::AssertTensorShape(node_translations, { node_count, 3 });
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorDevice(node_translations, device);

	o3c::AssertTensorShape(node_rotations, { node_count, 3, 3 });
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorDevice(node_rotations, device);

	o3c::AssertTensorShape(layer_decimation_radii, { layer_count });
	o3c::AssertTensorDtype(layer_decimation_radii, o3c::Float32);
	o3c::AssertTensorDevice(layer_decimation_radii, device);

	// input / output prep
	edge_residuals = o3c::Tensor({edge_count * 3}, o3c::Float32, device);
	auto edge_residual_data = edge_residuals.GetDataPtr<float>();

	auto edge_data = edges.GetDataPtr<int32_t>();
	auto edge_layer_index_data = edge_layer_indices.GetDataPtr<int8_t>();
	auto node_position_data = node_positions.GetDataPtr<float>();
	auto node_translation_data = node_translations.GetDataPtr<float>();
	auto node_rotation_data = node_rotations.GetDataPtr<float>();
	auto layer_decimation_radius_data = layer_decimation_radii.GetDataPtr<float>();

	// actual computation
	o3c::ParallelFor(
			device,
			edge_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_edge) {
				int32_t node_i = edge_data[i_edge * 2];
				int32_t node_j = edge_data[i_edge * 2 + 1];
				Eigen::Map<const Eigen::Vector3f> node_i_position(node_position_data + node_i * 3);
				Eigen::Map<const Eigen::Vector3f> node_j_position(node_position_data + node_j * 3);
				Eigen::Map<const Eigen::Vector3f> node_i_translation(node_translation_data + node_i * 3);
				Eigen::Map<const Eigen::Vector3f> node_j_translation(node_translation_data + node_j * 3);
				Eigen::Map<const core::kernel::Matrix3f> node_i_rotation(node_rotation_data + node_i * 9);
				float edge_weight = layer_decimation_radius_data[edge_layer_index_data[node_j]];
				Eigen::Map<Eigen::Vector3f> edge_residual(edge_residual_data + i_edge * 3);
				edge_residual = regularization_weight * edge_weight *
				                ((node_i_position + node_i_translation) - (node_j_position + node_j_translation) -
				                 node_i_rotation * (node_i_position - node_j_position));
			}
	);
}


template<open3d::core::Device::DeviceType TDevice>
void ComputeArapResiduals_VariableCoverageWeight(
		open3d::core::Tensor& edge_residuals,
		const open3d::core::Tensor& edges,
		const open3d::core::Tensor& node_positions,
		const open3d::core::Tensor& node_coverage_weights,
		const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& node_rotations,
		float regularization_weight
) {
	// counts & checks
	int64_t edge_count = edges.GetLength();
	int64_t node_count = node_positions.GetLength();

	o3c::Device device = edges.GetDevice();

	o3c::AssertTensorShape(edges, { edge_count, 2 });
	o3c::AssertTensorDtype(edges, o3c::Int32);

	o3c::AssertTensorShape(node_positions, { node_count, 3 });
	o3c::AssertTensorDtype(node_positions, o3c::Float32);
	o3c::AssertTensorDevice(node_positions, device);

	o3c::AssertTensorShape(node_coverage_weights, { node_count });
	o3c::AssertTensorDtype(node_coverage_weights, o3c::Float32);
	o3c::AssertTensorDevice(node_coverage_weights, device);

	o3c::AssertTensorShape(node_translations, { node_count, 3 });
	o3c::AssertTensorDtype(node_translations, o3c::Float32);
	o3c::AssertTensorDevice(node_translations, device);

	o3c::AssertTensorShape(node_rotations, { node_count, 3, 3 });
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);
	o3c::AssertTensorDevice(node_rotations, device);

	// input / output prep
	edge_residuals = o3c::Tensor({edge_count * 3}, o3c::Float32, device);
	auto edge_residual_data = edge_residuals.GetDataPtr<float>();

	auto edge_data = edges.GetDataPtr<int32_t>();
	auto node_weight_data = node_coverage_weights.GetDataPtr<float>();
	auto node_position_data = node_positions.GetDataPtr<float>();
	auto node_translation_data = node_translations.GetDataPtr<float>();
	auto node_rotation_data = node_rotations.GetDataPtr<float>();

	// actual computation
	o3c::ParallelFor(
			device,
			edge_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_edge) {
				int32_t node_i = edge_data[i_edge * 2];
				int32_t node_j = edge_data[i_edge * 2 + 1];
				Eigen::Map<const Eigen::Vector3f> node_i_position(node_position_data + node_i * 3);
				Eigen::Map<const Eigen::Vector3f> node_j_position(node_position_data + node_j * 3);
				Eigen::Map<const Eigen::Vector3f> node_i_translation(node_translation_data + node_i * 3);
				Eigen::Map<const Eigen::Vector3f> node_j_translation(node_translation_data + node_j * 3);
				Eigen::Map<const core::kernel::Matrix3f> node_i_rotation(node_rotation_data + node_i * 9);
				float node_i_weight = node_weight_data[node_i];
				float node_j_weight = node_weight_data[node_j];
				float edge_weight = fmaxf(node_i_weight, node_j_weight);
				Eigen::Map<Eigen::Vector3f> edge_residual(edge_residual_data + i_edge * 3);
				edge_residual = regularization_weight * edge_weight *
				                ((node_i_position + node_i_translation) - (node_j_position + node_j_translation) -
				                 node_i_rotation * (node_i_position - node_j_position));
			}
	);
}


} // namespace nnrt::alignment::kernel