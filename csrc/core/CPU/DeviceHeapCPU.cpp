//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/28/21.
//  Copyright (c) 2021 Gregory Kramida
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
#include <vector>
#include <algorithm>
#include "DeviceHeapCPU.h"



namespace o3c = open3d::core;

namespace nnrt::core {


template
class DeviceHeap<o3c::Device::DeviceType::CPU, KeyValuePair<float, int32_t>,
		decltype(MinHeapKeyCompare<KeyValuePair<float, int32_t>>)>;
template
class DeviceHeap<o3c::Device::DeviceType::CPU, KeyValuePair<float, int32_t>,
		decltype(MaxHeapKeyCompare<KeyValuePair<float, int32_t>>)>;
template
class DeviceHeap<o3c::Device::DeviceType::CPU, DistanceIndexPair<float, int32_t>,
		decltype(MinHeapKeyCompare<DistanceIndexPair<float, int32_t>>)>;
template
class DeviceHeap<o3c::Device::DeviceType::CPU, DistanceIndexPair<float, int32_t>,
		decltype(MaxHeapKeyCompare<DistanceIndexPair<float, int32_t>>)>;


} // namespace nnrt::core