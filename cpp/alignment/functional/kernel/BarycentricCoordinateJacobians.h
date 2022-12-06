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
#include "alignment/functional/kernel/MathTypedefs.h"


namespace nnrt::alignment::functional::kernel {

template<typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix2x3f CameraToNdcSpaceProjectionJacobian (){

}

template<rendering::functional::kernel::FrontFaceVertexOrder TVertexOrder = rendering::functional::kernel::CounterClockWise,
		typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> SignedFaceParallelogramAreaJacobianWrtNdcVertices(
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2
) {
	if (TVertexOrder == rendering::functional::kernel::CounterClockWise) {
		const Eigen::Vector2f dArea_dVertex0(vertex1.y() - vertex2.y(), vertex2.x - vertex1.x);
		const Eigen::Vector2f dArea_dVertex1(vertex2.y() - vertex0.y(), vertex0.x - vertex2.x);
		const Eigen::Vector2f dArea_dVertex2(vertex0.y() - vertex1.y(), vertex1.x - vertex0.x);
		return make_tuple(dArea_dVertex0, dArea_dVertex1, dArea_dVertex2);
	} else {
		const Eigen::Vector2f dArea_dVertex0(vertex2.y() - vertex1.y(), vertex1.x - vertex2.x);
		const Eigen::Vector2f dArea_dVertex1(vertex0.y() - vertex2.y(), vertex2.x - vertex0.x);
		const Eigen::Vector2f dArea_dVertex2(vertex1.y() - vertex0.y(), vertex0.x - vertex1.x);
		return make_tuple(dArea_dVertex0, dArea_dVertex1, dArea_dVertex2);
	}
}

template<rendering::functional::kernel::FrontFaceVertexOrder TVertexOrder = rendering::functional::kernel::CounterClockWise,
		typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f> SignedSubFaceParallelogramAreaJacobianWrtPointAndNdcVertices(
		const TPoint& point,
		const TVertex& vertex0,
		const TVertex& vertex1
) {
	if (TVertexOrder == rendering::functional::kernel::CounterClockWise) {
		const Eigen::Vector2f dArea_dVertex1(vertex1.y() - point.y(), point.x - vertex1.x);
		const Eigen::Vector2f dArea_dVertex2(point.y() - vertex0.y(), vertex0.x - point.x);
		return make_tuple(dArea_dVertex1, dArea_dVertex2);
	} else {
		const Eigen::Vector2f dArea_dVertex1(point.y() - vertex1.y(), vertex1.x - point.x);
		const Eigen::Vector2f dArea_dVertex2(vertex0.y() - point.y(), point.x - vertex0.x);
		return make_tuple(dArea_dVertex1, dArea_dVertex2);
	}
}

template<rendering::functional::kernel::FrontFaceVertexOrder TVertexOrder = rendering::functional::kernel::CounterClockWise, typename TPoint,
		typename TVertex, typename TVector3>
NNRT_DEVICE_WHEN_CUDACC
inline nnrt_tuple<Matrix3x2f,Matrix3x2f,Matrix3x2f>
BarycentricCoordinateJacobianWrtNdcVertices(const TPoint& intersection_point,
                                            const TVertex& vertex0,
                                            const TVertex& vertex1,
                                            const TVertex& vertex2,
                                            const TVector3 distorted_barycentric_coordinates) {
	// "p" stands for "parallelogram"
	const float face_parallelogram_area =
			rendering::functional::kernel::
			SignedParallelogramArea<TVertex, TVertex, rendering::functional::kernel::CounterClockWise>(vertex0, vertex1, vertex2)
			+ K_EPSILON;
	const float face_parallelogram_area_squared = face_parallelogram_area * face_parallelogram_area;
	const nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> d_face_area_wrt_vertices =
			SignedFaceParallelogramAreaJacobianWrtNdcVertices<TVertexOrder>(vertex0, vertex1, vertex2);

	// "p_area" stands for parallelogram_area
	const float sub_face0_p_area = distorted_barycentric_coordinates(0) * face_parallelogram_area;
	const float sub_face1_p_area = distorted_barycentric_coordinates(1) * face_parallelogram_area;
	const float sub_face2_p_area = distorted_barycentric_coordinates(2) * face_parallelogram_area;
	// const float sub_face_p_areas[] = {sub_face0_p_area, sub_face1_p_area, sub_face2_p_area};

	nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face0_p_area_wrt_vertices_1_2 =
			SignedSubFaceParallelogramAreaJacobianWrtPointAndNdcVertices(intersection_point, vertex1, vertex2);
	nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face1_p_area_wrt_vertices_2_0 =
			SignedSubFaceParallelogramAreaJacobianWrtPointAndNdcVertices(intersection_point, vertex2, vertex0);
	nnrt_tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face2_p_area_wrt_vertices_0_1 =
			SignedSubFaceParallelogramAreaJacobianWrtPointAndNdcVertices(intersection_point, vertex0, vertex1);

	Matrix3x2f d_distorted_coords_wrt_vertex0;
	Matrix3x2f d_distorted_coords_wrt_vertex1;
	Matrix3x2f d_distorted_coords_wrt_vertex2; // = Matrix3x2f::Zero(); -- how to start with zeros for Eigen

	// For the below jacobian, in code comments, we use ρ ("rho") for barycentric coordinates, and p for
	// projected normalized (Normalized Device Coordinate, or NDC space) triangle vertex coordinates.
	// i.e. if A_i is sub face signed parallelogram area (SPA) opposite vertex i and A_F is SPA of the whole triangular face,
	// The quotient rule works for all entries of the Jacobian (with {i, j, k} being {0, 1, 2} in any fixed order):
	//
	// ∂ρ_i/∂[p_i p_j p_k] = (A_F(∂A_i/∂[p_i p_j p_k]) - A_i(∂A_F/∂[p_i p_j p_k])) / A_F^2
	//
	// ∂ρ0/∂p0 , uses sub_face0 & vertex0
	d_distorted_coords_wrt_vertex0.row(0) =
			(-sub_face0_p_area * get<0>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;
	// ∂ρ1/∂p0 , uses sub_face1 & vertex0
	d_distorted_coords_wrt_vertex0.row(1) =
			(face_parallelogram_area * get<1>(d_sub_face1_p_area_wrt_vertices_2_0) - sub_face1_p_area * get<0>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared ;
	// ∂ρ2/∂p0 , uses sub_face2 & vertex0
	d_distorted_coords_wrt_vertex0.row(2) =
			(face_parallelogram_area * get<0>(d_sub_face2_p_area_wrt_vertices_0_1) - sub_face2_p_area * get<0>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;

	// ∂ρ0/∂p1 , uses sub_face0 & vertex1
	d_distorted_coords_wrt_vertex1.row(0) =
			(face_parallelogram_area * get<0>(d_sub_face0_p_area_wrt_vertices_1_2) - sub_face0_p_area * get<1>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;
	// ∂ρ1/∂p1 , uses sub_face1 & vertex1
	d_distorted_coords_wrt_vertex1.row(1) =
			(-sub_face1_p_area * get<1>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;
	// ∂ρ2/∂p1 , uses sub_face2 & vertex1
	d_distorted_coords_wrt_vertex1.row(2) =
			(face_parallelogram_area * get<1>(d_sub_face2_p_area_wrt_vertices_0_1) - sub_face2_p_area * get<1>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;

	// ∂ρ0/∂p2 , uses sub_face0 & vertex2
	d_distorted_coords_wrt_vertex2.row(0) =
			(face_parallelogram_area * get<1>(d_sub_face0_p_area_wrt_vertices_1_2) - sub_face0_p_area * get<2>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;
	// ∂ρ1/∂p2 , uses sub_face1 & vertex2
	d_distorted_coords_wrt_vertex2.row(1) =
			(face_parallelogram_area * get<0>(d_sub_face1_p_area_wrt_vertices_2_0) - sub_face1_p_area * get<2>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared ;
	// ∂ρ2/∂p2 , uses sub_face2 & vertex2
	d_distorted_coords_wrt_vertex2.row(2) =
			(-sub_face2_p_area * get<2>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;
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

} // namespace nnrt::alignment::functional::kernel