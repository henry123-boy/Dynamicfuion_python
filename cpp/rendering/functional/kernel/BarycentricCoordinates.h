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

// third-party includes
#include <Eigen/Dense>

// local includes
#include "core/platform_independence/Qualifiers.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/functional/kernel/FrontFaceVertexOrder.h"


namespace nnrt::rendering::functional::kernel {

/**
 * Returns signed area of the parallelogram given by two vectors (i.e. their exterior product, or wedge):
 * [point - vertex0] and [vertex0 - vertex1] (in the clockwise front face vertex order case)
 * [point - vertex0] and [vertex1 - vertex0] (in the counter-clockwise front face vertex order case)
 * Sign of the output also determines whether the normal of the triangle (defined by point and vertices) points toward (+) or away (-)
 * from the camera.
 */
template<FrontFaceVertexOrder TVertexOrder = ClockWise, typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline float SignedParallelogramArea(
		const TPoint& point, // can be a point of ray intersection or simply another face vertex
		const TVertex& vertex0, // face vertex
		const TVertex& vertex1 // face vertex
) {
	if (TVertexOrder == ClockWise) {
		return (point.x() - vertex0.x()) * (vertex0.y() - vertex1.y()) - (point.y() - vertex0.y()) * (vertex0.x() - vertex1.x());
	} else {
		return (point.x() - vertex0.x()) * (vertex1.y() - vertex0.y()) - (point.y() - vertex0.y()) * (vertex1.x() - vertex0.x());
	}
}


template<FrontFaceVertexOrder TVertexOrder = ClockWise, typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f BarycentricCoordinates(
		const TPoint& point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2
) {
	const float face_parallelogram_area = SignedParallelogramArea<ClockWise>(vertex0, vertex1, vertex2) + K_EPSILON;
	return {
			SignedParallelogramArea<ClockWise>(point, vertex1, vertex2) / face_parallelogram_area, // A_0 / A_f
			SignedParallelogramArea<ClockWise>(point, vertex2, vertex0) / face_parallelogram_area, // A_1 / A_f
			SignedParallelogramArea<ClockWise>(point, vertex0, vertex1) / face_parallelogram_area  // A_2 / A_f
	};
}

template<FrontFaceVertexOrder TVertexOrder = ClockWise, typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f BarycentricCoordinates_PreserveAreas(
		float& face_parallelogram_area,
		Eigen::Vector3f& sub_face_areas,
		const TPoint& point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2
) {
	face_parallelogram_area = SignedParallelogramArea<ClockWise>(vertex0, vertex1, vertex2) + K_EPSILON;
	sub_face_areas = Eigen::Vector3f(
			SignedParallelogramArea<ClockWise>(point, vertex1, vertex2),
			SignedParallelogramArea<ClockWise>(point, vertex2, vertex0),
			SignedParallelogramArea<ClockWise>(point, vertex0, vertex1)
	);
	return {
			sub_face_areas(0) / face_parallelogram_area, // A_0 / A_f
			sub_face_areas(1) / face_parallelogram_area, // A_1 / A_f
			sub_face_areas(2) / face_parallelogram_area  // A_2 / A_f
	};
}


NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f PerspectiveCorrectBarycentricCoordinates(
		const Eigen::Vector3f& distorted_barycentric_coordinates,
		const float vertex0z,
		const float vertex1z,
		const float vertex2z
) {
	const float coord0_numerator = distorted_barycentric_coordinates(0) * vertex1z * vertex2z;
	const float coord1_numerator = vertex0z * distorted_barycentric_coordinates(1) * vertex2z;
	const float coord2_numerator = vertex0z * vertex1z * distorted_barycentric_coordinates(2);
	const float denominator = fmaxf(coord0_numerator + coord1_numerator + coord2_numerator, K_EPSILON);
	return {coord0_numerator / denominator,
	        coord1_numerator / denominator,
	        coord2_numerator / denominator};
}


NNRT_DEVICE_WHEN_CUDACC
inline Eigen::Vector3f ClipBarycentricCoordinates(const Eigen::Vector3f& unclipped_barycentric_coordinates) {
	// Clip lower bound only
	Eigen::Vector3f clipped(
			fmaxf(unclipped_barycentric_coordinates.x(), 0.f),
			fmaxf(unclipped_barycentric_coordinates.y(), 0.f),
			fmaxf(unclipped_barycentric_coordinates.z(), 0.f)
	);
	clipped.normalize();
	return clipped;
}

} // namespace nnrt::rendering::functional::kernel
