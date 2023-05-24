//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/25/22.
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
#include "GeometrySampling.h"
#include <open3d/core/TensorCheck.h>
#include "geometry/functional/kernel/GeometrySampling.h"
#include <open3d/core/hashmap/HashSet.h>
#include "core/KdTree.h"

namespace o3c = open3d::core;

namespace nnrt::geometry::functional {


open3d::core::Tensor
MeanGridDownsample3dPoints(const open3d::core::Tensor& points, float grid_cell_size, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(points, { points.GetLength(), 3 });


	o3c::Tensor downsampled_points;
	functional::kernel::sampling::GridMeanDownsamplePoints(downsampled_points, points, grid_cell_size, hash_backend);
	return downsampled_points;
}


open3d::core::Tensor
ClosestToGridMeanSubsample3dPoints(const open3d::core::Tensor& points, float grid_cell_size, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(points, { points.GetLength(), 3 });
	o3c::Tensor downsampled_points;
	functional::kernel::sampling::GridMeanDownsamplePoints(downsampled_points, points, grid_cell_size, hash_backend);
	core::KdTree index(points);
	o3c::Tensor nearest_indices, squared_distances;
	index.FindKNearestToPoints(nearest_indices, squared_distances, downsampled_points, 4, true);


}



open3d::core::Tensor
FastMeanRadiusDownsample3dPoints(const open3d::core::Tensor& original_points, float radius, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(original_points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(original_points, { original_points.GetLength(), 3 });


	o3c::Tensor downsampled_points;
	functional::kernel::sampling::FastMeanRadiusDownsamplePoints(downsampled_points, original_points, radius, hash_backend);
	return downsampled_points;
}

open3d::core::Tensor
MedianGridSubsample3dPoints(const open3d::core::Tensor& points, float grid_cell_size, const open3d::core::HashBackendType& hash_backend) {
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(points, { points.GetLength(), 3 });
	o3c::Tensor sample;
	functional::kernel::sampling::GridMedianSubsample3dPoints(sample, points, grid_cell_size, hash_backend);
	return sample;
}


std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MedianGridSubsample3dPointsWithBinInfo(
		const open3d::core::Tensor& points,
		float grid_cell_size,
		open3d::core::Dtype bin_node_index_dtype,
		const open3d::core::HashBackendType& hash_backend
) {
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(points, { points.GetLength(), 3 });
	o3c::Tensor sample, other_bin_point_indices;
	functional::kernel::sampling::GridMedianSubsample3dPointsWithBinInfo(
			sample, other_bin_point_indices, points, grid_cell_size,
			hash_backend, bin_node_index_dtype);
	return std::make_tuple(sample, other_bin_point_indices);
}

open3d::core::Tensor
FastMedianRadiusSubsample3dPoints(const open3d::core::Tensor& points, float radius, const open3d::core::HashBackendType& hash_backend_type) {
	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
	o3c::AssertTensorShape(points, { points.GetLength(), 3 });
	o3c::Tensor sample;
	functional::kernel::sampling::FastMedianRadiusSubsample3dPoints(sample, points, radius, hash_backend_type);
	return sample;
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor>
RadiusSubsampleGraph(const open3d::core::Tensor& vertices, const open3d::core::Tensor& edges, float radius) {
	o3c::Tensor sample, resampled_edges;
	functional::kernel::sampling::RadiusSubsampleGraph(sample, resampled_edges, vertices, edges, radius);
	return std::make_tuple(sample, resampled_edges);
}



} // namespace nnrt::geometry::functional
