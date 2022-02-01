//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/2/22.
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

#include <cstdint>
#include <open3d/core/Blob.h>
#include "core/PlatformIndependence.h"

namespace nnrt::core::kernel::kdtree {

open3d::core::Blob BlobToDevice(const open3d::core::Blob& node_data, int64_t byte_count, const open3d::core::Device& device);

NNRT_DEVICE_WHEN_CUDACC
inline int32_t GetLeftChildIndex(int32_t parent_index){
	return 2 * parent_index + 1;
}

NNRT_DEVICE_WHEN_CUDACC
inline int32_t GetRightChildIndex(int32_t parent_index){
	return 2 * parent_index + 2;
}

NNRT_DEVICE_WHEN_CUDACC
inline int32_t GetParentIndex(int32_t child_index){
	return (child_index-1) / 2;
}

inline int32_t FindBalancedTreeIndexLength(const int32_t point_count, int& level_count) {
	int32_t count = 0;
	level_count = 0;
	for (int32_t range_start_index = 0, range_length = 1;
	     range_start_index < point_count;
	     range_start_index += range_length, range_length *= 2) {
		level_count += 1;
		count += range_length;
	}
	return count;
}

inline int32_t FindBalancedTreeIndexLength(const int32_t point_count) {
	int level_count;
	return FindBalancedTreeIndexLength(point_count, level_count);
}

template<unsigned int p>
int constexpr IntKnownPower(const int x) {
	if constexpr (p == 0) return 1;
	if constexpr (p == 1) return x;

	int tmp = IntKnownPower<p / 2>(x);
	if constexpr ((p % 2) == 0) { return tmp * tmp; }
	else { return x * tmp * tmp; }
}

static int IntPower(int x, unsigned int p) {
	if (p == 0) return 1;
	if (p == 1) return x;

	int tmp = IntPower(x, p / 2);
	if (p % 2 == 0) return tmp * tmp;
	else return x * tmp * tmp;
}

} // namespace nnrt::core::kernel::kdtree