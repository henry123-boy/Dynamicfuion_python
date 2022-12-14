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
#include "core/PlatformIndependentQualifiers.h"
#include "core/PlatformIndependentTuple.h"
#include "core/PlatformIndependentArray.h"
#include "rendering/functional/kernel/FrontFaceVertexOrder.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/functional/kernel/BarycentricCoordinates.h"
#include "alignment/functional/kernel/MathTypedefs.h"
#include "alignment/functional/kernel/ProjectionJacobians.h"


namespace o3tgk = open3d::t::geometry::kernel;

namespace nnrt::alignment::functional::kernel {

typedef rendering::functional::kernel::FrontFaceVertexOrder VertexOrder;

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename T2dVertex>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> Jacobian_SignedFaceParallelogramAreaWrt2dVertices(
		const T2dVertex& vertex0,
		const T2dVertex& vertex1,
		const T2dVertex& vertex2
) {
	if (TVertexOrder == rendering::functional::kernel::ClockWise) {
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

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TNdcIntersectionPoint, typename TNdcVertex>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Eigen::Vector2f, Eigen::Vector2f> Jacobian_SignedSubFaceParallelogramAreaWrtIntersectionPointAndNdcVertices(
		const TNdcIntersectionPoint& point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1
) {
	if (TVertexOrder == rendering::functional::kernel::ClockWise) {
		const Eigen::Vector2f dArea_dVertex1(vertex1.y() - point.y(), point.x - vertex1.x);
		const Eigen::Vector2f dArea_dVertex2(point.y() - vertex0.y(), vertex0.x - point.x);
		return make_tuple(dArea_dVertex1, dArea_dVertex2);
	} else {
		const Eigen::Vector2f dArea_dVertex1(point.y() - vertex1.y(), vertex1.x - point.x);
		const Eigen::Vector2f dArea_dVertex2(vertex0.y() - point.y(), point.x - vertex0.x);
		return make_tuple(dArea_dVertex1, dArea_dVertex2);
	}
}

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TNdcRayPoint, typename TNdcVertex, typename TBarycentricCoordinateVector,
		typename TGetFaceSubArea, typename TGetFaceParallelogramArea>
NNRT_DEVICE_WHEN_CUDACC
inline array<Matrix3x2f, 3>
Jacobian_BarycentricCoordinateWrtNdcVertices_Generic(
		const TNdcRayPoint& ray_point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1,
		const TNdcVertex& vertex2,
		TGetFaceSubArea&& get_face_sub_area,
		TGetFaceParallelogramArea&& get_face_parallelogram_area
) {
	const float face_parallelogram_area = get_face_parallelogram_area();
	const float face_parallelogram_area_squared = face_parallelogram_area * face_parallelogram_area;
	const tuple<Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f> d_face_area_wrt_vertices =
			Jacobian_SignedFaceParallelogramAreaWrt2dVertices<TVertexOrder>(vertex0, vertex1, vertex2);

	// "p_area" stands for parallelogram_area
	const float sub_face0_p_area = get_face_sub_area(0, face_parallelogram_area);
	const float sub_face1_p_area = get_face_sub_area(1, face_parallelogram_area);
	const float sub_face2_p_area = get_face_sub_area(2, face_parallelogram_area);

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

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TNdcRayPoint, typename TNdcVertex, typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline array<Matrix3x2f, 3>
Jacobian_BarycentricCoordinateWrtNdcVertices(
		const TNdcRayPoint& ray_point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1,
		const TNdcVertex& vertex2,
		const TBarycentricCoordinateVector distorted_barycentric_coordinates
) {
	return Jacobian_BarycentricCoordinateWrtNdcVertices_Generic(
			ray_point, vertex0, vertex1, vertex2,
			[&distorted_barycentric_coordinates](int i_sub_face, float face_parallelogram_area) {
				return distorted_barycentric_coordinates(i_sub_face) * face_parallelogram_area;
			},
			[&vertex0, &vertex1, &vertex2]() {
				return rendering::functional::kernel::
				       SignedParallelogramArea<TNdcVertex, TNdcVertex, rendering::functional::kernel::ClockWise>(vertex0, vertex1, vertex2)
				       + K_EPSILON;
			}
	);
}

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TNdcRayPoint, typename TNdcVertex>
NNRT_DEVICE_WHEN_CUDACC
inline array<Matrix3x2f, 3>
Jacobian_BarycentricCoordinateWrtNdcVertices(
		const TNdcRayPoint& ray_point,
		const TNdcVertex& vertex0,
		const TNdcVertex& vertex1,
		const TNdcVertex& vertex2,
		const float face_parallelogram_area,
		const Eigen::Vector3f& sub_face_parallelogram_areas
) {
	return Jacobian_BarycentricCoordinateWrtNdcVertices_Generic(
			ray_point, vertex0, vertex1, vertex2,
			[&sub_face_parallelogram_areas](int i_sub_face, float face_parallelogram_area) {
				return sub_face_parallelogram_areas(i_sub_face);
			},
			[&face_parallelogram_area]() {
				return face_parallelogram_area;
			}
	);
}


template<typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline tuple<Matrix3f, Matrix3f> Jacobian_PerspectiveCorrectBarycentricCoordinateWrtDistortedAndZ(
		TBarycentricCoordinateVector distorted_barycentric_coordinates,
		const float vertex0z,
		const float vertex1z,
		const float vertex2z
) {
	// ∂ρ0z1z2/∂p0 and also ∂g/∂p0, where g is the denominator in the forward-pass perspective-correction formula
	const float v1z_x_v2z = vertex1z * vertex2z;
	// ∂z0ρ1z2/∂p1 and also ∂g/∂p1, where g is the denominator
	const float v0z_x_v2z = vertex0z * vertex2z;
	// ∂z0z1ρ2/∂p2 and also ∂g/∂p2, where g is the denominator
	const float v0z_x_v1z = vertex0z * vertex1z;

	const float coord0_numerator = distorted_barycentric_coordinates(0) * v1z_x_v2z;
	const float coord1_numerator = distorted_barycentric_coordinates(1) * v0z_x_v2z;
	const float coord2_numerator = distorted_barycentric_coordinates(2) * v0z_x_v1z;
	const float denominator = fmaxf(coord0_numerator + coord1_numerator + coord2_numerator, K_EPSILON);
	const float denominator_squared = denominator * denominator;

	Matrix3f partial_coords_wrt_distorted;
	//@formatter:off
	partial_coords_wrt_distorted <<
	        (denominator - coord0_numerator) * v1z_x_v2z,
			-coord0_numerator * v0z_x_v2z,
			-coord0_numerator * v0z_x_v1z,

			-coord1_numerator * v1z_x_v2z,
			(denominator - coord1_numerator) * v0z_x_v2z,
			-coord1_numerator * v0z_x_v1z,

			-coord2_numerator * v1z_x_v2z,
			-coord2_numerator * v0z_x_v2z,
			(denominator - coord2_numerator) * v0z_x_v1z;
	//@formatter:on

	const float partial_denominator_wrt_z0 = distorted_barycentric_coordinates(1) * vertex2z + vertex1z * distorted_barycentric_coordinates(2);
	const float partial_denominator_wrt_z1 = distorted_barycentric_coordinates(0) * vertex2z + vertex0z * distorted_barycentric_coordinates(2);
	const float partial_denominator_wrt_z2 = distorted_barycentric_coordinates(0) * vertex1z + vertex0z * distorted_barycentric_coordinates(1);

	Eigen::Matrix3f partial_coords_wrt_z;
	//@formatter:off
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
	//@formatter:on
	partial_coords_wrt_z /= denominator_squared;

	return make_tuple(partial_coords_wrt_distorted, partial_coords_wrt_z);
}


template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TVertex, typename TPoint,
		typename TComputeJacobianBarycentricCoordinatesWrtNdcVertices,
		typename TApplyPerspectiveCorrectionJacobian>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix3x9f Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_Generic(
		const TPoint& ray_point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		const o3tgk::TransformIndexer& perspective_projection,
		TComputeJacobianBarycentricCoordinatesWrtNdcVertices&& compute_jacobian_barycentric_coordinates_wrt_ndc_vertices,
		TApplyPerspectiveCorrectionJacobian&& apply_corrections) {
	TVertex vertices_camera_space[] = {vertex0, vertex1, vertex2};
	Eigen::Vector2f ndc_vertex0, ndc_vertex1, ndc_vertex2;
	Eigen::Vector2f vertices_ndc[] = {ndc_vertex0, ndc_vertex1, ndc_vertex2};
	//TODO: test whether it makes sense to run this in multiple kernels, i.e. precompute per-pixel per-vertex operations in separate kernels in
	// order to utilize 3X number of threads.
	for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
		perspective_projection.Project(
				vertices_camera_space[i_vertex].x(), vertices_camera_space[i_vertex].y(), vertices_camera_space[i_vertex].z(),
				&vertices_ndc[i_vertex].x(), &vertices_ndc[i_vertex].y());
	}

	array<Matrix3x2f, 3> d_barycentric_coordinates_d_ndc =
			compute_jacobian_barycentric_coordinates_wrt_ndc_vertices(
					ndc_vertex0, ndc_vertex1, ndc_vertex2
			);

	float ndc_focal_coefficient_x, ndc_focal_coefficient_y;
	perspective_projection.GetFocalLength(&ndc_focal_coefficient_x, &ndc_focal_coefficient_y);
	//TODO: see above
	// Eigen::Matrix3f d_barycentric_coordinates_d_vertex0, d_barycentric_coordinates_d_vertex1, d_barycentric_coordinates_d_vertex2;
	// Eigen::Matrix3f d_barycentric_coordinates_d_vertices[] = {d_barycentric_coordinates_d_vertex0, d_barycentric_coordinates_d_vertex1,
	//                                                           d_barycentric_coordinates_d_vertex2};
	Matrix3x9f d_barycentric_coordinates_d_vertices;

	for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
		d_barycentric_coordinates_d_vertices.block<3, 3>(0, i_vertex * 3) =
				d_barycentric_coordinates_d_ndc[i_vertex] *
				CameraToNdcSpaceProjectionJacobian(ndc_focal_coefficient_x, ndc_focal_coefficient_y,
				                                   vertices_camera_space[i_vertex]);
	}

	apply_corrections(d_barycentric_coordinates_d_vertices);

	return d_barycentric_coordinates_d_vertices;
}

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TVertex, typename TPoint,
		typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix3x9f Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_WithPerspectiveCorrection(
		const TPoint& ray_point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		const o3tgk::TransformIndexer& perspective_projection) {
	Eigen::Vector3f distorted_barycentric_coordinates;
	return Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_Generic(
			ray_point, vertex0, vertex1, vertex2, perspective_projection,
			[&ray_point, &distorted_barycentric_coordinates](const Eigen::Vector2f& ndc_vertex0,
			                                                 const Eigen::Vector2f& ndc_vertex1,
			                                                 const Eigen::Vector2f& ndc_vertex2) {
				float face_parallelogram_area;
				Eigen::Vector3f sub_face_parallelogram_areas;
				distorted_barycentric_coordinates = rendering::functional::kernel::BarycentricCoordinates_PreserveAreas<TVertexOrder>(
						face_parallelogram_area, sub_face_parallelogram_areas,
						ray_point, ndc_vertex0, ndc_vertex1, ndc_vertex2
				);
				return Jacobian_BarycentricCoordinateWrtNdcVertices<TVertexOrder>(ray_point, ndc_vertex0, ndc_vertex1, ndc_vertex2,
				                                                                  face_parallelogram_area, sub_face_parallelogram_areas);
			},
			[&distorted_barycentric_coordinates, &vertex0, &vertex1, &vertex2](Matrix3x9f& d_barycentric_coordinates_d_vertices) {
				auto [d_perspective_corrected_d_distorted, d_perspective_corrected_d_z] =
						Jacobian_PerspectiveCorrectBarycentricCoordinateWrtDistortedAndZ(distorted_barycentric_coordinates, vertex0.z(), vertex1.z(),
						                                                                 vertex2.z());
				d_barycentric_coordinates_d_vertices = d_perspective_corrected_d_distorted * d_barycentric_coordinates_d_vertices;
				for (int i_coordinate = 0; i_coordinate < 3; i_coordinate++) {
					for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
						d_barycentric_coordinates_d_vertices(i_coordinate, i_vertex * 3 + 2) += d_perspective_corrected_d_z(i_coordinate, i_vertex);
					}
				}
			}

	);

}

template<VertexOrder TVertexOrder = VertexOrder::ClockWise, typename TVertex, typename TPoint,
		typename TBarycentricCoordinateVector>
NNRT_DEVICE_WHEN_CUDACC
inline Matrix3x9f Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_WithoutPerspectiveCorrection(
		const TPoint& ray_point,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		const o3tgk::TransformIndexer& perspective_projection,
		const TBarycentricCoordinateVector& distorted_barycentric_coordinates) {
	return Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_Generic(
			ray_point, vertex0, vertex1, vertex2, perspective_projection,
			[&ray_point, &distorted_barycentric_coordinates](const Eigen::Vector2f& ndc_vertex0,
			                                                 const Eigen::Vector2f& ndc_vertex1,
			                                                 const Eigen::Vector2f& ndc_vertex2) {
				return Jacobian_BarycentricCoordinateWrtNdcVertices<TVertexOrder>(ray_point, ndc_vertex0, ndc_vertex1, ndc_vertex2, distorted_barycentric_coordinates);
			},
			[](Matrix3x9f& d_barycentric_coordinates_d_vertices){}
	);
}

} // namespace nnrt::alignment::functional::kernel


