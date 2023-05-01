//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 1/26/23.
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
#include <open3d/core/Tensor.h>

// local includes
#include "test_main.hpp"

// code being tested
#include "core/linalg/Rodrigues.h"


namespace o3c = open3d::core;

void TestRodrigues(const o3c::Device& device) {
	o3c::Tensor axis_angle_vectors(std::vector<float>{
			0.16959796, 0.84415948, 0.63673575,
			-0.41038024, -0.41038024, 1.53155991,
			-0.35169471, -1.30855167, 1.67691329
	}, {3, 3}, o3c::Float32, device);

	o3c::Tensor expected_matrices(std::vector<float>{
			0.49240388, -0.45682599, 0.74084306,
			0.58682409, 0.80287234, 0.10504046,
			-0.64278761, 0.38302222, 0.66341395,

			0., -0.8660254, -0.5,
			1., 0., 0.,
			0., -0.5, 0.8660254,

			-0.51100229, -0.49471846, -0.70294402,
			0.80211293, 0.01955087, -0.59685225,
			0.30901699, -0.86883336, 0.38682953,
	}, {3, 3, 3}, o3c::Float32, device);

	o3c::Tensor matrices = nnrt::core::linalg::AxisAngleVectorsToMatricesRodrigues(axis_angle_vectors);

	REQUIRE(expected_matrices.AllClose(matrices,1e-3,1e-7));
}


TEST_CASE("Test Rodrigues - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRodrigues(device);
}

TEST_CASE("Test Rodrigues - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRodrigues(device);
}