//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/28/22.
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
// stdlib includes

// third-party includes
#include <Eigen/Dense>

// local includes
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentTuple.h"
#include "rendering/functional/kernel/FrontFaceVertexOrder.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/functional/kernel/BarycentricCoordinates.h"

namespace nnrt::rendering::functional::kernel {

template<FrontFaceVertexOrder TVertexOrder = CounterClockWise, typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> SignedParallelogramAreaJacobianWrtRaySpaceVertices(
		const TPoint& point, // can be a point of ray intersection or simply another face vertex
		const TVertex& vertex0, // face vertex
		const TVertex& vertex1 // face vertex
) {

	if (TVertexOrder == CounterClockWise) {
		const Eigen::Vector2f dArea_dPoint(vertex0.y() - vertex1.y(), vertex1.x - vertex0.x);
		const Eigen::Vector2f dArea_dVertex0(vertex1.y() - point.y(), point.x - vertex1.x);
		const Eigen::Vector2f dArea_dVertex1(point.y() - vertex0.y(), vertex0.x - point.x);
		return make_tuple(dArea_dPoint, dArea_dVertex0, dArea_dVertex1);
	} else {
		const Eigen::Vector2f dArea_dPoint(vertex1.y() - vertex0.y(), vertex0.x - vertex1.x);
		const Eigen::Vector2f dArea_dVertex0(point.y() - vertex1.y(), vertex1.x - point.x);
		const Eigen::Vector2f dArea_dVertex1(vertex0.y() - point.y(), point.x - vertex0.x);
		return make_tuple(dArea_dPoint, dArea_dVertex0, dArea_dVertex1);
	}
}

template<FrontFaceVertexOrder TVertexOrder = CounterClockWise, typename TPoint, typename TVertex, typename TVector3>
NNRT_DEVICE_WHEN_CUDACC
inline void BarycentricCoordinateJacobianWrtRaySpaceVertices(const TPoint& intersectionPoint,
															 const TVertex& vertex0,
															 const TVertex& vertex1,
															 const TVertex& vertex2,
															 const TVector3 distorted_barycentric_coordinates){
	const float face_parallelogram_area = SignedParallelogramArea<TVertex, TVertex, CounterClockWise>(vertex0, vertex1, vertex2) + K_EPSILON;
	const float face_parallelogram_area_squared = face_parallelogram_area * face_parallelogram_area;
	const float triangle0_area = distorted_barycentric_coordinates(0) * face_parallelogram_area;
	const float triangle1_area = distorted_barycentric_coordinates(1) * face_parallelogram_area;
	const float triangle2_area = distorted_barycentric_coordinates(2) * face_parallelogram_area;
	const nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> d_face_area_wrt_vertices =
			SignedParallelogramAreaJacobianWrtRaySpaceVertices<TVertexOrder>(vertex0, vertex1, vertex2);

	auto d_area_


}

NNRT_DEVICE_WHEN_CUDACC
inline nnrt_tuple<Eigen::Vector3f, Eigen::Matrix3f> PerspectiveCorrectBarycentricCoordinateJacobianWrtDistortedAndZ(
		const Eigen::Vector3f& distorted_barycentric_coordinates,
		const float vertex0z,
		const float vertex1z,
		const float vertex2z
) {
	const float partial_coord0_numerator_wrt_distorted0 = vertex1z * vertex2z;
	const float partial_coord1_numerator_wrt_distorted1 = vertex0z * vertex2z;
	const float partial_coord2_numerator_wrt_distorted2 = vertex0z * vertex1z;

	const float coord0_numerator = distorted_barycentric_coordinates(0) * partial_coord0_numerator_wrt_distorted0;
	const float coord1_numerator = distorted_barycentric_coordinates(1) * partial_coord1_numerator_wrt_distorted1;
	const float coord2_numerator = distorted_barycentric_coordinates(2) * partial_coord2_numerator_wrt_distorted2;
	const float denominator = FloatMax(coord0_numerator + coord1_numerator + coord2_numerator, K_EPSILON);
	const float denominator_squared = denominator * denominator;

	Eigen::Vector3f partial_coords_wrt_distorted(
			(denominator - coord0_numerator) * partial_coord0_numerator_wrt_distorted0 / denominator_squared,
			(denominator - coord1_numerator) * partial_coord1_numerator_wrt_distorted1 / denominator_squared,
			(denominator - coord2_numerator) * partial_coord2_numerator_wrt_distorted2 / denominator_squared
	);
	const float partial_denominator_wrt_z0 = distorted_barycentric_coordinates(1) * vertex2z + vertex1z * distorted_barycentric_coordinates(2);
	const float partial_denominator_wrt_z1 = distorted_barycentric_coordinates(0) * vertex2z + vertex0z * distorted_barycentric_coordinates(2);
	const float partial_denominator_wrt_z2 = distorted_barycentric_coordinates(0) * vertex1z + vertex0z * distorted_barycentric_coordinates(1);

	Eigen::Matrix3f partial_coords_wrt_z;
	partial_coords_wrt_z <<
	                     -coord0_numerator * partial_denominator_wrt_z0,
			denominator * distorted_barycentric_coordinates(0) * vertex2z - coord0_numerator * partial_denominator_wrt_z1,
			denominator * distorted_barycentric_coordinates(0) * vertex1z - coord0_numerator * partial_denominator_wrt_z2,

			denominator * distorted_barycentric_coordinates(1) * vertex2z - coord1_numerator * partial_denominator_wrt_z0,
			-coord1_numerator * partial_denominator_wrt_z1,
			denominator * distorted_barycentric_coordinates(1) * vertex0z - coord1_numerator * partial_denominator_wrt_z2,

			denominator * distorted_barycentric_coordinates(2) * vertex1z - coord2_numerator * partial_denominator_wrt_z0,
			denominator * distorted_barycentric_coordinates(2) * vertex0z - coord2_numerator * partial_denominator_wrt_z1,
			-coord2_numerator * partial_denominator_wrt_z2;
	partial_coords_wrt_z /= denominator_squared;

	return make_tuple(partial_coords_wrt_distorted, partial_coords_wrt_z);
}

} // namespace nnrt::rendering::functional::kernel