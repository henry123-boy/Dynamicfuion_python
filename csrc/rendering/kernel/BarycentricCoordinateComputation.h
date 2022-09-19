//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/5/22.
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

#include <Eigen/Dense>
#include "core/PlatformIndependence.h"
#include "RasterizationConstants.h"


namespace nnrt::rendering::kernel {

// Determines whether the given point is on the right side of a 2D line segment given by end points vertex0 and vertex1.
// Returns signed area of the parallelogram given by the vectors [point - vertex0] and [vertex1 - vertex0]
template<typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline float ComputeSignedParallelogramArea(
		const TPoint& point,
		const TVertex& vertex0,
		const TVertex& vertex1
) {
	return (point.x() - vertex0.x()) * (vertex0.y() - vertex1.y()) - (point.y() - vertex0.y()) * (vertex0.x() - vertex1.x());
}

template<typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f ComputeBarycentricCoordinates(
		const TPoint& point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2
) {
	const float area = ComputeSignedParallelogramArea(vertex2, vertex0, vertex1) + K_EPSILON;
	return {
			ComputeSignedParallelogramArea(point, vertex1, vertex2) / area,
			ComputeSignedParallelogramArea(point, vertex2, vertex0) / area,
			ComputeSignedParallelogramArea(point, vertex0, vertex1) / area
	};
}

NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f PerspectiveCorrectBarycentricCoordinates(
		const Eigen::Vector3f& distorted_barycentric_coordinates,
		const float vertex0z,
		const float vertex1z,
		const float vertex2z
) {
	const float coord0_numerator = distorted_barycentric_coordinates.x() * vertex1z * vertex2z;
	const float coord1_numerator = vertex0z * distorted_barycentric_coordinates.y() * vertex2z;
	const float coord2_numerator = vertex0z * vertex1z * distorted_barycentric_coordinates.z();
	const float denominator = FloatMax(coord0_numerator + coord1_numerator + coord2_numerator, K_EPSILON);
	return {coord0_numerator / denominator,
	        coord1_numerator / denominator,
	        coord2_numerator / denominator};
}

NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f ClipBarycentricCoordinates(const Eigen::Vector3f& unclipped_barycentric_coordinates) {
	// Clip lower bound only
	Eigen::Vector3f clipped(
			FloatMax(unclipped_barycentric_coordinates.x(), 0.f),
			FloatMax(unclipped_barycentric_coordinates.y(), 0.f),
			FloatMax(unclipped_barycentric_coordinates.z(), 0.f)
	);
	clipped.normalize();
	return clipped;
}

} // namespace nnrt::rendering::kernel
