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
#include <open3d/t/geometry/kernel/GeometryIndexer.h>


// local includes
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentTuple.h"
#include "core/PlatformIndependentArray.h"
#include "rendering/functional/kernel/FrontFaceVertexOrder.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/functional/kernel/BarycentricCoordinates.h"
#include "alignment/functional/kernel/MathTypedefs.h"
#include "alignment/functional/kernel/ProjectionJacobians.h"



namespace o3tgk = open3d::t::geometry::kernel;

namespace nnrt::alignment::functional::kernel {

typedef rendering::functional::kernel::FrontFaceVertexOrder VertexOrder;

template<VertexOrder TVertexOrder = VertexOrder::CounterClockWise, typename T2dVertex>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> Jacobian_SignedFaceParallelogramAreaWrt2dVertices(
		const T2dVertex& vertex0,
		const T2dVertex& vertex1,
		const T2dVertex& vertex2
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

template<VertexOrder TVertexOrder = VertexOrder::CounterClockWise, typename TNdcIntersectionPoint, typename TNdcVertex>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Eigen::Vector2f, Eigen::Vector2f> Jacobian_SignedSubFaceParallelogramAreaWrtIntersectionPointAndNdcVertices(
		const TNdcIntersectionPoint& point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1
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

template<VertexOrder TVertexOrder = VertexOrder::CounterClockWise, typename TNdcRayPoint, typename TNdcVertex, typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
//inline tuple<Matrix3x2f, Matrix3x2f, Matrix3x2f>
inline array<Matrix3x2f, 3>
Jacobian_BarycentricCoordinateWrtNdcVertices(
		const TNdcRayPoint& ray_point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1,
		const TNdcVertex& vertex2,
		const TBarycentricCoordinateVector distorted_barycentric_coordinates
) {
	// "p" stands for "parallelogram"
	const float face_parallelogram_area =
			rendering::functional::kernel::
			SignedParallelogramArea<TNdcVertex, TNdcVertex, rendering::functional::kernel::CounterClockWise>(vertex0, vertex1, vertex2)
			+ K_EPSILON;
	const float face_parallelogram_area_squared = face_parallelogram_area * face_parallelogram_area;
	const tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> d_face_area_wrt_vertices =
			Jacobian_SignedFaceParallelogramAreaWrt2dVertices<TVertexOrder>(vertex0, vertex1, vertex2);

	// "p_area" stands for parallelogram_area
	const float sub_face0_p_area = distorted_barycentric_coordinates(0) * face_parallelogram_area;
	const float sub_face1_p_area = distorted_barycentric_coordinates(1) * face_parallelogram_area;
	const float sub_face2_p_area = distorted_barycentric_coordinates(2) * face_parallelogram_area;
	// const float sub_face_p_areas[] = {sub_face0_p_area, sub_face1_p_area, sub_face2_p_area};

	tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face0_p_area_wrt_vertices_1_2 =
			Jacobian_SignedSubFaceParallelogramAreaWrtIntersectionPointAndNdcVertices(ray_point, vertex1, vertex2);
	tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face1_p_area_wrt_vertices_2_0 =
			Jacobian_SignedSubFaceParallelogramAreaWrtIntersectionPointAndNdcVertices(ray_point, vertex2, vertex0);
	tuple<Eigen::Vector2f, Eigen::Vector2f> d_sub_face2_p_area_wrt_vertices_0_1 =
			Jacobian_SignedSubFaceParallelogramAreaWrtIntersectionPointAndNdcVertices(ray_point, vertex0, vertex1);

	Matrix3x2f d_distorted_coords_wrt_vertex0, d_distorted_coords_wrt_vertex1, d_distorted_coords_wrt_vertex2;
	//Note: = Matrix3x2f::Zero(); -- how to initialize zero matrix for Eigen

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
			/ face_parallelogram_area_squared;
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
			/ face_parallelogram_area_squared;
	// ∂ρ2/∂p2 , uses sub_face2 & vertex2
	d_distorted_coords_wrt_vertex2.row(2) =
			(-sub_face2_p_area * get<2>(d_face_area_wrt_vertices))
			/ face_parallelogram_area_squared;

	return array<Matrix3x2f, 3>({d_distorted_coords_wrt_vertex0, d_distorted_coords_wrt_vertex1, d_distorted_coords_wrt_vertex2});
}


template<typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Eigen::RowVector3f, Eigen::Matrix3f> Jacobian_PerspectiveCorrectBarycentricCoordinateWrtDistortedAndZ(
		TBarycentricCoordinateVector distorted_barycentric_coordinates,
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

	//TODO: this should be a row-major matrix instead! Fix math!
	Eigen::RowVector3f partial_coords_wrt_distorted(
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


template<bool TWithPerspectiveCorrection, VertexOrder TVertexOrder = VertexOrder::CounterClockWise, typename TVertex, typename TPoint,
		typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix3x9f Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices(
		const TPoint& ray_point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		const o3tgk::TransformIndexer& perspective_projection,
		const TBarycentricCoordinateVector& distorted_barycentric_coordinates) {
	TVertex vertices_camera_space[] = {vertex0, vertex1, vertex2};
	Eigen::Vector2f ndc_vertex0, ndc_vertex1, ndc_vertex2;
	Eigen::Vector2f vertices_ndc[] = {ndc_vertex0, ndc_vertex1, ndc_vertex2};
	//TODO: test whether it makes sense to run this in multiple kernels, i.e. precompute per-pixel per-vertex operations in separate kernels in
	// order to utilize 3X number of threads.
	for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
		perspective_projection.Project(vertices_camera_space[i_vertex].x(), vertices_camera_space[i_vertex].y(),
		                               vertices_camera_space[i_vertex].z(),
		                               &vertices_ndc[i_vertex].x(), &vertices_ndc[i_vertex].y());
	}
	tuple<Matrix3x2f, Matrix3x2f, Matrix3x2f> d_barycentric_coordinates_d_ndc_tuple =
			Jacobian_BarycentricCoordinateWrtNdcVertices<TVertexOrder>(ray_point, ndc_vertex0, ndc_vertex1, ndc_vertex2,
			                                                           distorted_barycentric_coordinates);
	Matrix3x2f d_barycentric_coordinates_d_ndc[] = {get<0>(d_barycentric_coordinates_d_ndc_tuple),
	                                                get<1>(d_barycentric_coordinates_d_ndc_tuple),
	                                                get<2>(d_barycentric_coordinates_d_ndc_tuple)};
	float ndc_focal_coefficient_x, ndc_focal_coefficient_y;
	perspective_projection.GetFocalLength(&ndc_focal_coefficient_x, &ndc_focal_coefficient_y);
	//TODO: see above
	// Eigen::Matrix3f d_barycentric_coordinates_d_vertex0, d_barycentric_coordinates_d_vertex1, d_barycentric_coordinates_d_vertex2;
	// Eigen::Matrix3f d_barycentric_coordinates_d_vertices[] = {d_barycentric_coordinates_d_vertex0, d_barycentric_coordinates_d_vertex1,
	//                                                           d_barycentric_coordinates_d_vertex2};
	Matrix3x9f d_barycentric_coordinates_d_vertices;

	for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
		d_barycentric_coordinates_d_vertices.block<3,3>(0, i_vertex*3) =
				d_barycentric_coordinates_d_ndc[i_vertex] *
				CameraToNdcSpaceProjectionJacobian(ndc_focal_coefficient_x, ndc_focal_coefficient_y,
				                                   vertices_camera_space[i_vertex]);
	}

	if(TWithPerspectiveCorrection){
		tuple<Eigen::RowVector3f, Eigen::Matrix3f> d_perspective_corrected_d_distorted_and_z =
				Jacobian_PerspectiveCorrectBarycentricCoordinateWrtDistortedAndZ(distorted_barycentric_coordinates, vertex0.z(), vertex1.z(), vertex2.z());
		Eigen::RowVector3f d_perspective_corrected_d_distorted = get<0>(d_perspective_corrected_d_distorted_and_z);

	}

	// return make_tuple(d_barycentric_coordinates_d_vertex0, d_barycentric_coordinates_d_vertex1, d_barycentric_coordinates_d_vertex2);
	return d_barycentric_coordinates_d_vertices;
}


} // namespace nnrt::alignment::functional::kernel


