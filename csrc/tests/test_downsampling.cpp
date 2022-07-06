//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/9/22.
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
#include <fmt/ranges.h>

#include "test_main.hpp"

#include <open3d/core/Tensor.h>
#include <geometry/DownsamplePoints.h>

#include <Eigen/Dense>

using namespace nnrt;
namespace o3c = open3d::core;


void SortPointsVector(std::vector<float>& point_data) {
	std::vector<Eigen::Vector3f> downsampled_points_eigen;
	auto point_count = point_data.size() / 3;
	for (int i_point = 0; i_point < point_count; i_point++) {
		downsampled_points_eigen.emplace_back(point_data[i_point * 3],
		                                      point_data[i_point * 3 + 1],
		                                      point_data[i_point * 3 + 2]);
	}
	// sort by X coordinate
	std::sort(downsampled_points_eigen.begin(), downsampled_points_eigen.end(),
	          [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> bool {
		          return a.x() < b.x();
	          });
	point_data.clear();
	for (const Eigen::Vector3f& point: downsampled_points_eigen) {
		point_data.push_back(point.x());
		point_data.push_back(point.y());
		point_data.push_back(point.z());
	}
}

template<typename TDownsample>
void TestGridDownsampling_Generic(const o3c::Device& device, TDownsample&& downsample) {
	std::vector<float> point_data = {3.09232593e+00, 3.80791020e+00, 0.00000000e+00, -1.05966675e+00,
	                                 -1.41309094e+00, 0.00000000e+00, -4.37189054e+00, 4.47028160e+00,
	                                 0.00000000e+00, -4.89017248e+00, 1.23748720e+00, 0.00000000e+00,
	                                 -2.22830486e+00, 1.38681889e+00, 0.00000000e+00, 2.83344054e+00,
	                                 3.08373499e+00, 0.00000000e+00, -2.21787024e+00, 1.95599473e+00,
	                                 0.00000000e+00, -2.44953656e+00, 4.53404427e+00, 0.00000000e+00,
	                                 1.83352754e-01, -4.20935774e+00, 0.00000000e+00, 1.96778488e+00,
	                                 -4.51444674e+00, 0.00000000e+00, -4.05896616e+00, -4.75339603e+00,
	                                 0.00000000e+00, -1.71987343e+00, -3.95355105e+00, 0.00000000e+00,
	                                 -4.20096684e+00, -6.37725174e-01, 0.00000000e+00, -1.86513603e+00,
	                                 2.48662806e+00, 0.00000000e+00, 3.72645259e+00, -3.78359628e+00,
	                                 0.00000000e+00, 1.39817727e+00, -2.40715075e+00, 0.00000000e+00,
	                                 -2.57892871e+00, -2.97164392e+00, 0.00000000e+00, 4.53065825e+00,
	                                 2.28084016e+00, 0.00000000e+00, -4.73043299e+00, 1.57850909e+00,
	                                 0.00000000e+00, -3.69062400e+00, 1.24971382e-03, 0.00000000e+00,
	                                 -3.13266039e+00, -4.63495207e+00, 0.00000000e+00, 1.12872040e+00,
	                                 -1.30202389e+00, 0.00000000e+00, -4.23765993e+00, -1.30425775e+00,
	                                 0.00000000e+00, -2.24193454e-01, 2.49021840e+00, 0.00000000e+00,
	                                 -4.82457113e+00, -3.56276965e+00, 0.00000000e+00, 1.71533477e+00,
	                                 4.98386478e+00, 0.00000000e+00, -3.44815803e+00, -3.99885511e+00,
	                                 0.00000000e+00, 1.25434265e-01, 2.28916669e+00, 0.00000000e+00,
	                                 1.40168875e-01, 2.65022850e+00, 0.00000000e+00, 3.76163363e+00,
	                                 3.85235727e-01, 0.00000000e+00};
	o3c::Tensor points(point_data, {30, 3}, o3c::Dtype::Float32, device);
	o3c::Tensor downsampled_points = downsample(points, 5.0);
	auto downsampled_points_data = downsampled_points.ToFlatVector<float>();
	SortPointsVector(downsampled_points_data);

	// ground truth computed via Python/NumPy
	std::vector<float> downsampled_points_gt_data = {-3.2952073f, -2.7228992f, 0.0f, -2.8721921f, 2.5174978f, 0.0f,
	                                                 1.6808975f, -3.2433152f, 0.0f, 2.3141422f, 2.7829971f, 0.0f};

	REQUIRE(std::equal(downsampled_points_data.begin(), downsampled_points_data.end(), downsampled_points_gt_data.begin(),
	                   [](float a, float b) { return a == Approx(b).margin(1e-6).epsilon(1e-12); }));
}

void TestGridDownsampling_PlainArray(const o3c::Device& device) {
	TestGridDownsampling_Generic(device, geometry::GridDownsample3DPoints_PlainBinArray);
}

TEST_CASE("Test Grid Downsampling - Plain Array - CPU") {
	auto device = o3c::Device("CPU:0");
	TestGridDownsampling_PlainArray(device);
}

TEST_CASE("Test Grid Downsampling - Plain Array - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestGridDownsampling_PlainArray(device);
}

void TestGridDownsampling_Hash(const o3c::Device& device) {
	TestGridDownsampling_Generic(device, [](o3c::Tensor& points, float grid_cell_size) {
		return geometry::GridDownsample3DPoints_BinHash(points, grid_cell_size);
	});
}

TEST_CASE("Test Grid Downsampling - Hash - CPU") {
	auto device = o3c::Device("CPU:0");
	TestGridDownsampling_Hash(device);
}

TEST_CASE("Test Grid Downsampling - Hash - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestGridDownsampling_Hash(device);
}
