//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/8/22.
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

// 3rd party
#include <open3d/core/Device.h>

// local
#include "core/platform_independence/Qualifiers.h"
#include "core/platform_independence/Atomics.h"
#include "../../../../3rd-party/Eigen/Eigen/Dense"


struct PointAggregationBin {
#ifdef __CUDACC__
	float x;
	float y;
	float z;
	int count;
#else
	std::atomic<float> x;
	std::atomic<float> y;
	std::atomic<float> z;
	std::atomic<int> count;
#endif
	template<open3d::core::Device::DeviceType DeviceType, typename TPoint>
	NNRT_DEVICE_WHEN_CUDACC
	void UpdateWithPointAndCount(const TPoint& point, int _count){
		x = point.x();
		y = point.y();
		z = point.z();
		this->count = _count;
	}
};