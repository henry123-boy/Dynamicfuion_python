//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/11/23.
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
// stdlib includes

// third-party includes

// local includes
#include "tests/test_atomic_counter_array/test_atomic_counter_array_impl.h"

template
void TestAtomicCounterArray<open3d::core::Device::DeviceType::CPU>(const open3d::core::Device& device);
