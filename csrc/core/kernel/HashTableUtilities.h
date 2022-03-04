//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/3/22.
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
//TODO: remove this header if unused
#include <open3d/core/Tensor.h>
#include <Eigen/Dense>

#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentAtomics.h"

namespace o3c = open3d::core;


namespace nnrt::core::kernel::hash_table {
template<open3d::core::Device::DeviceType DeviceType>
inline NNRT_DEVICE_WHEN_CUDACC
int HashCodeFromBinPosition(const Eigen::Vector3<int16_t>& block_pos, int32_t hash_mask) {
	return static_cast<int>((((uint) block_pos.x() * 73856093u) ^ ((uint) block_pos.y() * 19349669u) ^ ((uint) block_pos.z() * 83492791u)) &
	                        (uint) hash_mask);
}

template<open3d::core::Device::DeviceType DeviceType>
inline NNRT_DEVICE_WHEN_CUDACC
Eigen::Vector3<int16_t> DeterminePointsBinCoordinate(const Eigen::Map<Eigen::Vector3f>& point,
                                                     const Eigen::Vector3f& grid_center,
                                                     float bin_size) {
	Eigen::Vector3f point_after_offset = point - grid_center;
	return {
			static_cast<int16_t>(((point_after_offset.x() < 0) ? point_after_offset.x() - bin_size + 1 : point_after_offset.x()) / bin_size),
			static_cast<int16_t>(((point_after_offset.y() < 0) ? point_after_offset.y() - bin_size + 1 : point_after_offset.y()) / bin_size),
			static_cast<int16_t>(((point_after_offset.z() < 0) ? point_after_offset.z() - bin_size + 1 : point_after_offset.z()) / bin_size)
	};
}

/// Closest least power of 2 minus 1. Returns 0 if n = 0.
template<typename UInt, std::enable_if_t<std::is_unsigned<UInt>::value, int> = 0>
constexpr UInt clp2m1(UInt n, unsigned i = 1) noexcept { return i < sizeof(UInt) * 8 ? clp2m1(UInt(n | (n >> i)), i << 1) : n; }

/// Closest least power of 2 minus 1. Returns 0 if n <= 0.
template<typename Int, std::enable_if_t<std::is_integral<Int>::value && std::is_signed<Int>::value, int> = 0>
constexpr auto clp2m1(Int n) noexcept { return clp2m1(std::make_unsigned_t<Int>(n <= 0 ? 0 : n)); }

/// Closest least power of 2. Returns 2^N: 2^(N-1) < n <= 2^N. Returns 0 if n <= 0.
template<typename Int, std::enable_if_t<std::is_integral<Int>::value, int> = 0>
constexpr auto ClosestLeastPowerOf2(Int n) noexcept {
	return clp2m1(std::make_unsigned_t<Int>(n - 1)) + 1;
}

/** \brief
	A single bin in the hash table holding an average point.
*/
struct AveragePointHashBin {
	/** Position of the corner of the 8x8x8 volume, that identifies the entry. */
	Eigen::Vector3<int16_t> position;
	/** Offset in the excess list. */
	int offset;
	bool active;

	NNRT_DECLARE_ATOMIC(float, x);
	NNRT_DECLARE_ATOMIC(float, y);
	NNRT_DECLARE_ATOMIC(float, z);
	NNRT_DECLARE_ATOMIC(int, count);


};


template<open3d::core::Device::DeviceType DeviceType>
NNRT_DEVICE_WHEN_CUDACC
inline int FindHashCodeAt(const AveragePointHashBin* hash_table, const Eigen::Vector3<int16_t>& at,
                          int32_t hash_mask, int32_t ordered_list_size) {
	int hash = HashCodeFromBinPosition<DeviceType>(at, hash_mask);
	while (true) {
		const AveragePointHashBin& hash_entry = hash_table[hash];

		if (hash_entry.position == at && hash_entry.active) {
			return hash;
		}

		if (hash_entry.offset < 1) break;
		hash = ordered_list_size + hash_entry.offset - 1;
	}
	return -1;
}
//
// template<open3d::core::Device::DeviceType DeviceType>
// NNRT_DEVICE_WHEN_CUDACC
// inline bool FindOrAllocateHashEntry(const Eigen::Vector3<int32_t>& hash_bin_position,
//                                     AveragePointHashBin*& result_entry,
//                                     AveragePointHashBin* hash_table,
//                                     int ordered_bin_count,
//                                     NNRT_ATOMIC_ARGUMENT(int) active_bin_count,
// 									NNRT_ATOMIC_ARGUMENT(int) last_free_excess_bin_index,
//                                     const int* excess_allocation_list,
//                                     int* active_hash_codes,
//                                     int& hash_code) {
// 	hash_code = HashCodeFromBinPosition<DeviceType>(hash_bin_position);
// 	AveragePointHashBin* hash_entry = hash_table + hash_code;
// 	if (hash_entry->position != hash_bin_position || !hash_entry->active) {
// 		bool add_to_excess_list = false;
// 		//search excess list only if there is no room in ordered part
// 		if (hash_entry->active) {
// 			while (hash_entry->offset >= 1) {
// 				hash_code = ordered_bin_count + hash_entry->offset - 1;
// 				hash_entry = hash_table + hash_code;
// 				if (hash_entry->position != hash_bin_position && !hash_entry->active) {
// 					result_entry = &hash_table[hash_code];
// 					return true;
// 				}
// 			}
// 			add_to_excess_list = true;
// 		}
// 		//still not found, allocate
// 		if (add_to_excess_list && last_free_excess_bin_index >= 0) {
// 			//there is room in excess bin list
// 			AveragePointHashBin& new_hash_entry = hash_table[hash_code];
// 			new_hash_entry.position = hash_bin_position;
// 			new_hash_entry.active = true;
// 			new_hash_entry.offset = 0;
// 			int excess_list_offset = excess_allocation_list[last_free_excess_bin_index];
// 			hash_table[hash_code].offset = excess_list_offset + 1; //connect to child
// 			hash_code = ordered_bin_count + excess_list_offset;
//
// 			result_entry = &hash_table[hash_code];
// 			last_free_excess_bin_index--;
// 			active_hash_codes[active_bin_count] = hash_code;
// 			NNRT_ATOMIC_ADD(active_bin_count, 1);
// 			return true;
// 		} else if (last_free_voxel_block_id >= 0) {
// 			//there is room in the voxel block array
// 			HashEntry new_hash_entry;
// 			new_hash_entry.pos = hash_bin_position;
// 			new_hash_entry.ptr = voxel_allocation_list[last_free_voxel_block_id];
// 			new_hash_entry.offset = 0;
// 			hash_table[hash_code] = new_hash_entry;
// 			result_entry = &hash_table[hash_code];
// 			last_free_voxel_block_id--;
// 			active_hash_codes[utilized_hash_code_count] = hash_code;
// 			utilized_hash_code_count++;
// 			return true;
// 		} else {
// 			return false;
// 		}
// 	} else {
// 		//HashEntry already exists, return the pointer to it
// 		result_entry = &hash_table[hash_code];
// 		return true;
// 	}
// }

} // namespace nnrt::core::kernel::hash_table