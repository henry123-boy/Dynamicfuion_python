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

namespace nnrt::rendering::kernel {
/*
 * A GridBitMask represents a boolean array of shape (height, width, items_per_block). We pack values into the bits of unsigned ints to enable
 * easy bit-setting via >> and << operators. A single unsigned int has the size of one word, four bytes, or 32 bits (word_bit_count).
 * Hence, to hold the entire mask we use height * width * (item_count_per_block / word_bit_count) unsigned ints. We want to store
 * GridBitMasks in block-shared memory, so we assume that the memory has already been allocated for it elsewhere.
 */
class GridBitMask {
public:
	__device__ GridBitMask(unsigned int* data, int gird_height_in_cells, int grid_width_in_cells, int item_count_per_block)
			: data(data), gird_height_in_cells(gird_height_in_cells), grid_width_in_cells(grid_width_in_cells),
			  word_count(item_count_per_block / word_bit_count) {
		if (data == nullptr || item_count_per_block % 32 != 0)
			asm("trap;"); // item_count_per_block should be a multiple of CUDA warp size, which is 32. The data cannot be a null pointer.
	}

	// Use all threads in the current block to clear all the bits
	__device__ void block_clear() {
		for (int i_word = threadIdx.x; i_word < gird_height_in_cells * grid_width_in_cells * word_count; i_word += blockDim.x) {
			data[i_word] = 0;
		}
		__syncthreads();
	}

	__device__ int get_word_index(int y, int x, int item_index) {
		return (y * grid_width_in_cells + x) * word_count + item_index / word_bit_count;
	}

	__device__ int get_bit_within_word_index(int item_index) {
		return item_index % word_bit_count;
	}

	// Set the single bit at (y, x, item_index) to one
	__device__ void set(int y, int x, int item_index) {
		int elem_idx = get_word_index(y, x, item_index);
		int bit_idx = get_bit_within_word_index(item_index);
		const unsigned int mask = 1U << bit_idx;
		atomicOr(data + elem_idx, mask);
	}

	// Set the single bit at (y, x, item_index) to zero
	__device__ void unset(int y, int x, int item_index) {
		int elem_idx = get_word_index(y, x, item_index);
		int bit_idx = get_bit_within_word_index(item_index);
		const unsigned int mask = ~(1U << bit_idx);
		atomicAnd(data + elem_idx, mask);
	}

	// Check whether the bit (y, x, d) is zero or one
	__device__ bool get(int y, int x, int item_index) {
		int elem_idx = get_word_index(y, x, item_index);
		int bit_idx = get_bit_within_word_index(item_index);
		return (data[elem_idx] >> bit_idx) & 1U;
	}

	// Compute the number of bits set to one for a grid cell (y, x)
	__device__ int count(int y, int x) {
		int total = 0;
		for (int i_word = 0; i_word < word_count; ++i_word) {
			int word_index = y * grid_width_in_cells * word_count + x * word_count + i_word;
			unsigned int word = data[word_index];
			total += __popc(word);
		}
		return total;
	}

private:
	unsigned int* data;
	static constexpr int word_bit_count = 8 * sizeof(unsigned int);
	int gird_height_in_cells, grid_width_in_cells, word_count;
};
} // namespace nnrt::rendering::kernel