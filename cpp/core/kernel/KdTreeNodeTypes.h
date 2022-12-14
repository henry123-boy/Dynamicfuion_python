//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/17/22.
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
#include "core/PlatformIndependentQualifiers.h"

namespace nnrt::core::kernel::kdtree {
struct KdTreeNode {
	int32_t point_index;
	uint8_t i_split_dimension;

	NNRT_DEVICE_WHEN_CUDACC
	bool Empty() const {
		return point_index == -1;
	}

	NNRT_DEVICE_WHEN_CUDACC
	void Clear() {
		point_index = -1;
	}

};

struct RangeNode {
	KdTreeNode node;
	int32_t range_start;
	int32_t range_end;
};
} // namespace nnrt::core::kernel::kdtree