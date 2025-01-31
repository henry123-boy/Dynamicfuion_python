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
// third-party
#include <open3d/core/Tensor.h>
#include <Eigen/Dense>
#include <set>

// test utilities
#include "test_main.hpp"
#include "tests/test_utils/test_utils.hpp"
#include "catch2/catch_approx.hpp"

// code being tested
#include "geometry/functional/GeometrySampling.h"
#include "geometry/functional/ComputeDistanceMatrix.h"
#include "core/functional/Masking.h"

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

inline
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Tensor2DfToEigenMatrix(const open3d::core::Tensor& tensor) {
	auto tensor_data = tensor.ToFlatVector<float>();
	assert(tensor.GetShape().size() == 2);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix_map(tensor_data.data(), tensor.GetShape(0),
	                                                                                             tensor.GetShape(1));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(matrix_map);
	return matrix;
}

inline
Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Tensor2DfDataToEigenMatrix(const std::vector<float>& tensor_data, int rows, int cols) {
	return {tensor_data.data(), rows, cols};
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

	std::vector<float> downsampled_points_gt_data = {-3.25127237f, -3.02558241f, 0.f,
	                                                 -2.96312902f, 2.23791466f, 0.f,
	                                                 1.68089758f, -3.24331508f, 0.f,
	                                                 2.31414232f, 2.78299729f, 0.f};


	REQUIRE(std::equal(downsampled_points_data.begin(), downsampled_points_data.end(), downsampled_points_gt_data.begin(),
	                   [](float a, float b) { return a == Catch::Approx(b).margin(1e-6).epsilon(1e-12); }));
}


void TestGridDownsampling_Hash(const o3c::Device& device) {
	TestGridDownsampling_Generic(device, [](o3c::Tensor& points, float grid_cell_size) {
		return geometry::functional::MeanGridDownsample3dPoints(points, grid_cell_size);
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



void TestRadiusDownsampling(const o3c::Device& device) {
	o3c::Tensor points = o3c::Tensor::Load(test::static_array_test_data_directory.ToString() + "/downsampling_source.npy");
	float downsampling_radius = 10.0;
	o3c::Tensor downsampled_points = geometry::functional::FastMeanRadiusDownsample3dPoints(points, 10.0);
	o3c::Tensor distance_matrix = geometry::functional::ComputeDistanceMatrix(downsampled_points, downsampled_points);
	core::functional::ReplaceValue(distance_matrix, 0.f, std::numeric_limits<float>::max());
	float min_distance = distance_matrix.Min({0,1}).ToFlatVector<float>()[0];
	REQUIRE(min_distance >= downsampling_radius);

}


TEST_CASE("Test Radius Downsampling - CPU") {
	auto device = o3c::Device("CPU:0");
	TestRadiusDownsampling(device);
}

TEST_CASE("Test Radius Downsampling - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestRadiusDownsampling(device);
}

void TestGridMedianDownsampling(const o3c::Device& device){
	std::vector<float> point_data = {
			1.36, 1.2, 1,
			1.41, 1.95, 1,
			1.76, 1.5, 1, // median for 1, 1, 1 grid cell
			2.46, 2.75, 1, // median for 2, 2, 2 grid cell
			2.61, 2.45, 1,
			2.26, 2.3, 1,
			3.51, 2.6, 1, // either of two points can be median in 3, 2, 1 grid cell
			3.56, 2.35, 1,
			3.66, 1.7, 1, // either of two points can be median in 3, 1, 1 grid cell
			3.11, 1.05, 1
	};
	o3c::Tensor points(point_data, {10, 3}, o3c::Dtype::Float32, device);
	o3c::Tensor sample = geometry::functional::MedianGridSubsample3dPoints(points, 1.0f);
	auto sample_data = sample.ToFlatVector<int64_t>();
	std::set<int64_t> sample_data_set(std::make_move_iterator(sample_data.begin()), std::make_move_iterator(sample_data.end()));

	REQUIRE(sample_data_set.find((int64_t)2) != sample_data_set.end());
	REQUIRE(sample_data_set.find((int64_t)4) != sample_data_set.end());
	REQUIRE((sample_data_set.find((int64_t)6) != sample_data_set.end() || sample_data_set.find((int64_t)7) != sample_data_set.end()));
	REQUIRE((sample_data_set.find((int64_t)8) != sample_data_set.end() || sample_data_set.find((int64_t)9) != sample_data_set.end()));

}

TEST_CASE("Test Median Grid Downsampling - CPU") {
	auto device = o3c::Device("CPU:0");
	TestGridMedianDownsampling(device);
}

TEST_CASE("Test Median Grid Downsampling - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestGridMedianDownsampling(device);
}

void TestClosestToMeanSampling(const o3c::Device& device){
	std::vector<float> point_data = {
			// 0-2, 0-2; mean: 1.01428571, 0.9, 1.
			0.15, 0.3, 1 , // 0
			1.25, 0.35, 1,
			0.5, 0.8, 1  , // 2 - closest to mean
			0.4, 1.7, 1  ,
			1.45, 1.65, 1,
			1.65, 1.15, 1,
			1.7, 0.35, 1 , // 6

			// 2-4, 0-2; mean: 3.08 , 1.045, 1.
			2.45, 0.35, 1, // 7
			3.1, 0.25, 1 , // 8
			3.7, 0.4, 1  ,
			3.8, 1.05, 1 ,
			3.8, 1.7, 1  ,
			3.2, 1.65, 1 ,
			3.2, 1.05, 1 , // 13 - closest to mean
			2.55, 1.3, 1 ,
			2.55, 1.75, 1,
			2.45, 0.95, 1 // 16
	};
	o3c::Tensor points(point_data, {17, 3}, o3c::Dtype::Float32, device);
	o3c::Tensor sample = geometry::functional::ClosestToGridMeanSubsample3dPoints(points, 2.0f);
	o3c::Tensor sample_gt(std::vector<int64_t>{2, 13}, {2}, o3c::Int64, device);
	REQUIRE(sample.AllEqual(sample_gt));
}

TEST_CASE("Test Closest To Grid Mean Sampling - CPU") {
	auto device = o3c::Device("CPU:0");
	TestClosestToMeanSampling(device);
}

TEST_CASE("Test Closest To Grid Mean Sampling - CUDA") {
	auto device = o3c::Device("CUDA:0");
	TestClosestToMeanSampling(device);
}