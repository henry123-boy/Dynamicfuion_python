//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/16/22.
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

// 3rd-party
#include <open3d/core/Tensor.h>
#include <open3d/core/CUDAUtils.h>

// local
#include "rendering/kernel/RasterizeMeshImpl.h"
#include "rendering/kernel/GridBitMask.cuh"

namespace nnrt::rendering::kernel {

#define NNRT_BITMASK_BLOCK_SIZE 512

template<unsigned int TChunkSize>
__global__
void GridBin2dBoundingBoxes_Kernel(
		int* bins,
		int* bin_face_counts,
		const float* bounding_boxes,
		const bool* boxes_to_skip_mask,
		const int bounding_box_count,
		const int bounding_box_count_per_batch,

		const int batch_count_per_thread,

		const int image_height,
		const int image_width,

		const int bin_count_y,
		const int bin_count_x,

		const int bin_side_length,
		const int bin_capacity,
		const float half_pixel_x,
		const float half_pixel_y
) {
	extern __shared__ char shared_memory_buffer[];
	const int block_size = static_cast<int>(blockDim.x);
	const int block_index = static_cast<int>(blockIdx.x);
	const int thread_index = static_cast<int>(threadIdx.x);

	GridBitMask block_overlap_registry((unsigned int*) shared_memory_buffer, bin_count_y, bin_count_x, block_size);
	const int bin_count = bin_count_y * bin_count_x;

	for (int i_batch = 0; i_batch < batch_count_per_thread; i_batch++) {

		block_overlap_registry.block_clear();

		const int first_batch_box_index = i_batch * bounding_box_count_per_batch;
		const int first_block_box_index = first_batch_box_index + block_size * block_index;
		const int i_box_in_block = thread_index;
		const int i_box = first_block_box_index + i_box_in_block;

		if (i_box < bounding_box_count && !boxes_to_skip_mask[i_box]) {
			const float x_min = bounding_boxes[0 * bounding_box_count + i_box];
			const float x_max = bounding_boxes[1 * bounding_box_count + i_box];
			const float y_min = bounding_boxes[2 * bounding_box_count + i_box];
			const float y_max = bounding_boxes[3 * bounding_box_count + i_box];

			// Brute-force search all bins for overlaps with bounding boxes
			for (int bin_y = 0; bin_y < bin_count_y; bin_y++) {
				const float bin_y_min = ImageSpaceToNormalizedCameraSpace(bin_y * bin_side_length, image_height, image_width) - half_pixel_y;
				const float bin_y_max =
						ImageSpaceToNormalizedCameraSpace((bin_y + 1) * bin_side_length - 1, image_height, image_width) + half_pixel_y;
				const bool y_overlap = (y_min <= bin_y_max) && (bin_y_min < y_max);

				if (y_overlap) {
					for (int bin_x = 0; bin_x < bin_count_x; bin_x++) {
						const float bin_x_min = ImageSpaceToNormalizedCameraSpace(bin_x * bin_side_length, image_width, image_height) - half_pixel_x;
						const float bin_x_max =
								ImageSpaceToNormalizedCameraSpace((bin_x + 1) * bin_side_length - 1, image_width, image_height) + half_pixel_x;
						const bool x_overlap = (x_min <= bin_x_max) && (bin_x_min < x_max);
						if (x_overlap) {
							// int32_t bin_index = bin_y * grid_height_bins + bin_x;
							// mark corresponding bit as "1" for overlap between grid & cell in the current block's registry.
							block_overlap_registry.set(bin_y, bin_x, i_box_in_block);
						}
					}
				}
			}
		}
		__syncthreads();

		// loop over bins and count up the total faces that overlap each
		for (int bin_index = thread_index; bin_index < bin_count; bin_index += block_size) {
			const int bin_y = bin_index / bin_count_x;
			const int bin_x = bin_index % bin_count_x;

			const int overlap_count = block_overlap_registry.count(bin_y, bin_x);

			/*
			 * Atomically increment the (global) number of boxes found in the current bin and gets the previous value of the counter;
			 * this effectively allocates space in the bin_faces array for the elems in the current chunk that fall into this bin.
			 */
			const int start = atomicAdd(bin_face_counts + bin_index, overlap_count);

			if (start + overlap_count > bin_capacity) {
				/*
				* The number of elems in this bin is so big that they won't fit. Print a warning using CUDA's printf.
				*/
				printf("Bin size was too small in the grid binning phase. This caused an overflow, meaning output may be incomplete. "
				       "To correct this, try increasing max_faces_per_bin, decreasing bin_size, or setting bin_size to 0 to use naive rasterization.\n");
				continue;
			}
			int bin_box_index = bin_index * bin_capacity;
			// Loop over bitmask and write active bits for this bin
			for (int i_box_in_block2 = 0; i_box_in_block2 < block_size; i_box_in_block2++) {
				if (block_overlap_registry.get(bin_y, bin_x, i_box_in_block2)) {
					bins[bin_box_index] = first_block_box_index + i_box_in_block2;
					bin_box_index++;
				}
			}
		}
		__syncthreads();
	} // end batch loop



}

template<>
void GridBin2dBoundingBoxes_Device<open3d::core::Device::DeviceType::CUDA>(
		open3d::core::Tensor& bins,
		const open3d::core::Tensor& bounding_boxes,
		const open3d::core::Tensor& boxes_to_skip_mask,
		const int image_height,
		const int image_width,
		const int bin_count_y,
		const int bin_count_x,
		const int bin_side_length,
		const int bin_capacity,
		const float half_pixel_x,
		const float half_pixel_y
) {

#define USE_BLOCK_BIT_MASK
#ifdef USE_BLOCK_BIT_MASK
	auto device = bounding_boxes.GetDevice();
	o3c::CUDAScopedDevice scoped_device(device);

	const auto bounding_box_count = static_cast<int>(bounding_boxes.GetShape(1));

	// we will always work on a single bit mask per CUDA block, since "shared memory" is only shared by threads within any given CUDA block.
	const int bits_per_byte = 8;
	const size_t shared_memory_per_block = bin_count_y * bin_count_x * NNRT_BITMASK_BLOCK_SIZE / bits_per_byte;
	// block count will dictate how many blocks (and, therefore, bitmasks) we can process in a single batch
	const size_t block_count = 64;
	const int batch_count_per_thread = static_cast<int>((bounding_box_count - 1) / (block_count * NNRT_BITMASK_BLOCK_SIZE) + 1);
	// corresponds to thread count in the whole CUDA grid
	const int bounding_box_count_per_block_batch = NNRT_BITMASK_BLOCK_SIZE * block_count;

	o3c::Tensor bin_face_counts = o3c::Tensor::Zeros({bin_count_y, bin_count_x}, o3c::Int32, device);

	GridBin2dBoundingBoxes_Kernel<NNRT_BITMASK_BLOCK_SIZE><<<block_count, NNRT_BITMASK_BLOCK_SIZE, shared_memory_per_block, o3c::cuda::GetStream()>>>(
			bins.GetDataPtr<int32_t>(),
			bin_face_counts.GetDataPtr<int32_t>(),
			bounding_boxes.GetDataPtr<float>(),
			boxes_to_skip_mask.GetDataPtr<bool>(),
			bounding_box_count,
			bounding_box_count_per_block_batch,

			batch_count_per_thread,

			image_height,
			image_width,

			bin_count_y,
			bin_count_x,

			bin_side_length,
			bin_capacity,
			half_pixel_x,
			half_pixel_y);

	o3c::OPEN3D_GET_LAST_CUDA_ERROR("GridBin2dBoundingBoxes_Device failed.");
#else
	auto bounding_box_count = bounding_boxes.GetShape(1);
	auto bin_data = bins.GetDataPtr<int32_t>();
	auto device = bounding_boxes.GetDevice();
	auto boxes_mask_data = boxes_to_skip_mask.GetDataPtr<bool>();
	auto bounding_box_data = bounding_boxes.GetDataPtr<float>();

	o3c::Tensor bin_face_count_tensor = o3c::Tensor::Zeros({bin_count_x,bin_count_y}, o3c::Int32, device);
	int* bin_face_counts = bin_face_count_tensor.GetDataPtr<int32_t>();
	o3c::ParallelFor(
			device, bounding_box_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				if (boxes_mask_data[workload_idx]) {
					return;
				}
				float x_min = bounding_box_data[0 * bounding_box_count + workload_idx];
				float x_max = bounding_box_data[1 * bounding_box_count + workload_idx];
				float y_min = bounding_box_data[2 * bounding_box_count + workload_idx];
				float y_max = bounding_box_data[3 * bounding_box_count + workload_idx];


				for (int bin_y = 0; bin_y < bin_count_y; bin_y++) {
					const float bin_y_min =
							ImageSpaceToNormalizedCameraSpace(bin_y * bin_side_length, image_height, image_width) - half_pixel_y;
					const float bin_y_max =
							ImageSpaceToNormalizedCameraSpace((bin_y + 1) * bin_side_length - 1, image_height, image_width) + half_pixel_y;
					const bool y_overlap = (y_min <= bin_y_max) && (bin_y_min < y_max);

					if (y_overlap) {
						for (int bin_x = 0; bin_x < bin_count_x; bin_x++) {

							const float bin_x_min =
									ImageSpaceToNormalizedCameraSpace(bin_x * bin_side_length, image_width, image_height) - half_pixel_x;
							const float bin_x_max =
									ImageSpaceToNormalizedCameraSpace((bin_x + 1) * bin_side_length - 1, image_width, image_height) + half_pixel_x;
							const bool x_overlap = (x_min <= bin_x_max) && (bin_x_min < x_max);

							if (x_overlap) {
								int32_t bin_index = bin_y * bin_count_x + bin_x;
								int32_t insertion_position = atomicAdd(bin_face_counts + bin_index, 1);

								// store active face index into the bin if they overlap spatially
								bin_data[bin_index * bin_capacity + insertion_position] = static_cast<int32_t>(workload_idx);
							}
						}
					}
				}
			}
	);
#endif
}

} // namespace nnrt::rendering::kernel