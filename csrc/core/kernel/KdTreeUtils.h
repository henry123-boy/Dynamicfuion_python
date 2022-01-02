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

namespace nnrt::core::kernel::kdtree {

inline int64_t FindBalancedTreeIndexLength(const int64_t point_count, int& level_count) {
	int64_t count = 0;
	level_count = 0;
	for (int64_t range_start_index = 0, range_length = 1;
	     range_start_index < point_count;
	     range_start_index += range_length, range_length *= 2) {
		level_count += 1;
		count += range_length;
	}
	return count;
}

template <unsigned int p>
int constexpr IntPower(const int x){
	if constexpr (p == 0) return 1;
	if constexpr (p == 1) return x;

	int tmp = IntPower<p / 2>(x);
	if constexpr ((p % 2) == 0) { return tmp * tmp; }
	else { return x * tmp * tmp; }
}

} // namespace nnrt::core::kernel::kdtree