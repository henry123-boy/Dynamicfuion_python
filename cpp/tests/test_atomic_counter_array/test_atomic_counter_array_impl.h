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
#pragma once
// stdlib includes

// third-party includes
#include "catch2/catch_test_macros.hpp"

// local includes
#include "tests/test_atomic_counter_array/test_atomic_counter_array.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/ParallelFor.h"
#include "core/platform_independence/Qualifiers.h"

template<open3d::core::Device::DeviceType TDeviceType>
void TestAtomicCounterArray(const open3d::core::Device& device) {
    int counters_count = 20;
    int64_t work_unit_count = 100;
    int increment = 2;

    REQUIRE(work_unit_count % counters_count == 0);


    nnrt::core::AtomicCounterArray counters(counters_count);

    nnrt::core::ParallelForMutable(
            device, work_unit_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t work_unit_index) mutable {
                int counter_index = work_unit_index % counters_count;
                counters.FetchAdd(counter_index, increment);
            }
    );

    open3d::core::Tensor final_counts = counters.AsTensor(true);
    open3d::core::Tensor expected_final_counts({counters_count}, open3d::core::Int32, device);
    int expected_final_count = (work_unit_count / counters_count) * increment;
    expected_final_counts.Fill(expected_final_count);

    REQUIRE(final_counts.AllClose(expected_final_counts));
}