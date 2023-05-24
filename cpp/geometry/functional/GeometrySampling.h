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
#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/HashMap.h>

namespace nnrt::geometry::functional {


/**
 * \brief average-downsample points within each grid cell of a uniform cubic grid
 * \param downsampled_points resulting points
 * \param points original points in open3d::core::Float32 format
 * \param grid_cell_size side length of each (cubic) grid cell
 * \param hash_backend hash backend to use for the operation
 * \return resulting point set as Nx3 open3d::core::Float32 tensor, stored on same device as the point input tensor
 */
open3d::core::Tensor MeanGridDownsample3dPoints(
		const open3d::core::Tensor& points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default
);

open3d::core::Tensor
ClosestToGridMeanSubsample3dPoints(
		const open3d::core::Tensor& points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default
);


/**
 * \brief average-downsample points such that resulting points are at least at `radius` apart.
 * \details does not necessarily produce a maximal downsampling set, but efficient since it uses two grid-averaging
 * downsampling iterations (with a 1/2 grid-cell offset in the second iteration) to produce the result.
 * \param downsampled_points resulting downsampled points
 * \param original_points original points in open3d::core::Float32 format
 * \param min_distance minimal distance between resulting points
 * \param hash_backend backend to use for hashing
 * \return resulting point set as Nx3 open3d::core::Float32 tensor, stored on same device as the point input tensor
 */
open3d::core::Tensor FastMeanRadiusDownsample3dPoints(
		const open3d::core::Tensor& original_points, float radius,
		const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default
);

/**
 * \brief median-sample points within each grid cell of a uniform cubic grid
 * \param points original points in open3d::core::Float32 format, tensor of size Nx3
 * \param grid_cell_size side length of each (cubic) grid cell
 * \param hash_backend hash backend to use for the operation
 * \return resulting point indices as N-sized open3d::core::Float32 tensor, stored on same device as the point input tensor
 */
//TODO write doc
open3d::core::Tensor
MedianGridSubsample3dPoints(
		const open3d::core::Tensor& points, float grid_cell_size,
		const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default
);

/**
 * \brief median-sample points within each grid cell of a uniform cubic grid and compute the difference sets for each cell
 * \param points original points in open3d::core::Float32 format, tensor of size Nx3
 * \param grid_cell_size side length of each (cubic) grid cell
 * \param hash_backend hash backend to use for the operation
 * \return returns <medians,bin_points>, where "medians" is an N-sized open3d::core::Float32 tensor containing the medians,
 * while bin_points is a -1-padded 2D array with indices of the non-sampled points remaining in each bin
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
MedianGridSubsample3dPointsWithBinInfo(
		const open3d::core::Tensor& points, float grid_cell_size,
		open3d::core::Dtype bin_node_index_dtype = open3d::core::Int32,
		const open3d::core::HashBackendType& hash_backend = open3d::core::HashBackendType::Default
);

/**
 * \brief median-downsample points such that resulting sample consists of points at least `radius` apart from one another.
 * \param points original points
 * \param radius minimum distance between selected points
 * \param hash_backend_type backend to use for hashing
 * \return resulting sample of point indexes, in a 1D Int64 tensor, stored on the same device as the original points
 */
open3d::core::Tensor
FastMedianRadiusSubsample3dPoints(
		const open3d::core::Tensor& points, float radius,
		const open3d::core::HashBackendType& hash_backend_type = open3d::core::HashBackendType::Default
);

/**
 * \brief (Untested, most likely suffers heavily from race conditions) sub-sample a dense graph (can potentially be used for construction of the
 * edge structure for embedded shape deformation using a planar graph)
 * \param vertices
 * \param edges
 * \param radius
 * \return
 */
std::tuple<open3d::core::Tensor, open3d::core::Tensor>
RadiusSubsampleGraph(const open3d::core::Tensor& vertices, const open3d::core::Tensor& edges, float radius);


} // nnrt::geometry::functional