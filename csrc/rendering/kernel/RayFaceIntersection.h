//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/6/22.
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

// 3rd party
#include <Eigen/Dense>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

// local
#include "rendering/kernel/CoordinateSystemConversions.h"
#include "rendering/kernel/BarycentricCoordinateUtilities.h"


namespace o3tgk = open3d::t::geometry::kernel;
namespace nnrt::rendering::kernel {
using t_face_index = int32_t;

struct RayFaceIntersection {
	float depth;
	int64_t face_index; // index of face
	float distance; // distance of intersection to face
	Eigen::Vector3f barycentric_coordinates;
};


// Determine whether the point (px, py) lies outside the face 2D bounding box while accounting for the blur radius.
template<typename TPoint, typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline bool PointOutsideFaceBoundingBox(
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		float blur_radius,
		const TPoint& point
) {
	const float x_min = FloatMin3(vertex0.x(), vertex1.x(), vertex2.x()) - blur_radius;
	const float x_max = FloatMax3(vertex0.x(), vertex1.x(), vertex2.x()) + blur_radius;

	const float y_min = FloatMin3(vertex0.y(), vertex1.y(), vertex2.y()) - blur_radius;
	const float y_max = FloatMax3(vertex0.y(), vertex1.y(), vertex2.y()) + blur_radius;

	const float z_min = FloatMin3(vertex0.z(), vertex1.z(), vertex2.z());

	// Faces with at least one vertex behind the camera won't render correctly
	// and should be removed or clipped before rasterizing
	const bool z_invalid = z_min < K_EPSILON;

	return (point.x() > x_max || point.x() < x_min || point.y() > y_max || point.y() < y_min || z_invalid);
}

template<typename TPoint, typename TSegmentEndpoint>
NNRT_DEVICE_WHEN_CUDACC
inline float PointSegmentSquareDistance(
		const TPoint& point,
		const TSegmentEndpoint& segment_vertex0,
		const TSegmentEndpoint& segment_vertex1
) {
	const TPoint segment = segment_vertex1 - segment_vertex0;
	const float squared_segment_length = segment.dot(segment);
	float ratio_of_closest_point_along_segment = segment.dot(point - segment_vertex0) / squared_segment_length;
	if (squared_segment_length <= K_EPSILON) {
		const TPoint vertex1_to_point = point - segment_vertex1;
		return vertex1_to_point.dot(vertex1_to_point);
	}
	ratio_of_closest_point_along_segment = FloatClampTo0To1(ratio_of_closest_point_along_segment);
	const TPoint closest_point_on_segment = segment_vertex0 + ratio_of_closest_point_along_segment * segment;
	const TPoint point_to_segment = (closest_point_on_segment - point);
	return point_to_segment.dot(point_to_segment); // squared distance
}


template<typename TPoint, typename TTriangleVertex>
NNRT_DEVICE_WHEN_CUDACC
inline float PointTriangleDistance(
		const TPoint& p,
		const TTriangleVertex& v0,
		const TTriangleVertex& v1,
		const TTriangleVertex& v2
) {
	// Compute distance to all three edges and return the minimum.
	const float e01_dist = PointSegmentSquareDistance(p, v0, v1);
	const float e02_dist = PointSegmentSquareDistance(p, v0, v2);
	const float e12_dist = PointSegmentSquareDistance(p, v1, v2);
	const float edge_dist = FloatMin3(e01_dist, e02_dist, e12_dist);
	return edge_dist;
}



/*
 * Assumes normalized camera-space coordinates for face vertices and pixel
 */
NNRT_DEVICE_WHEN_CUDACC
inline void UpdateQueueIfPixelInsideFace(
		const o3tgk::TArrayIndexer <t_face_index>& face_vertex_position_indexer,
		t_face_index i_face,
		RayFaceIntersection* queue, int queue_size,
		float queue_max_depth, int queue_max_depth_at,
		float blur_radius, const Eigen::Vector2f& pixel,
		const int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
) {
	auto face_vertices_data = face_vertex_position_indexer.GetDataPtr<float>(i_face);
	Eigen::Map<Eigen::Vector3f> face_vertex0(face_vertices_data);
	Eigen::Map<Eigen::Vector3f> face_vertex1(face_vertices_data + 3);
	Eigen::Map<Eigen::Vector3f> face_vertex2(face_vertices_data + 6);

	Eigen::Map<Eigen::Vector2f> face_vertex0_xy(face_vertices_data);
	Eigen::Map<Eigen::Vector2f> face_vertex1_xy(face_vertices_data + 3);
	Eigen::Map<Eigen::Vector2f> face_vertex2_xy(face_vertices_data + 6);

	const float face_area = ComputeSignedParallelogramArea(face_vertex0_xy, face_vertex1_xy, face_vertex2_xy);
	const bool is_back_face = face_area < 0.f;
	const bool zero_face_area = (face_area <= K_EPSILON && face_area >= -1.f * K_EPSILON);

	/* If any of these conditions are true:
	 * (1) the pixel is outside the triangle's bounding box in XY plane (allowing for blur radius),
	 * (2) the face is a collapsed triangle (area is zero), or
	 * (3) we're culling back-faces and the face is back-facing,
	 * do nothing and return.
	 */
	if (PointOutsideFaceBoundingBox(face_vertex0, face_vertex1, face_vertex2, FloatSquareRoot(blur_radius), pixel) ||
	    (cull_back_faces && is_back_face) || zero_face_area) {
		return;
	}

	Eigen::Vector3f barycentric_coordinates = ComputeBarycentricCoordinates(pixel, face_vertex0_xy, face_vertex1_xy, face_vertex2_xy);
	if (perspective_correct_barycentric_coordinates) {
		barycentric_coordinates =
				PerspectiveCorrectBarycentricCoordinates(barycentric_coordinates, face_vertex0.z(), face_vertex1.z(), face_vertex2.z());
	}
	Eigen::Vector3f barycentric_coordinates_clipped =
			clip_barycentric_coordinates ? ClipBarycentricCoordinates(barycentric_coordinates) : barycentric_coordinates;

	const float point_z = barycentric_coordinates_clipped.x() * face_vertex0.z() +
	                      barycentric_coordinates_clipped.y() * face_vertex1.z() +
	                      barycentric_coordinates_clipped.z() * face_vertex2.z();
	if (point_z < 0) {
		return; // Face is behind image plane.
	}
	const float point_face_distance = PointTriangleDistance(pixel, face_vertex0_xy, face_vertex1_xy, face_vertex2_xy);

	// Use the unclipped barycentric coordinates to determine if the point is inside the face.
	const bool inside = barycentric_coordinates.x() > 0.f && barycentric_coordinates.y() > 0.f && barycentric_coordinates.z() > 0.f;
	const float signed_point_face_distance = inside ? -point_face_distance : point_face_distance;

	// If pixel is both outside and farther away than the blur radius, exit
	if (!inside && point_face_distance >= blur_radius){
		return;
	}

	/* Handle the case where a face "f" partially behind the image plane is clipped to a quadrilateral and then split into two faces (t1, t2).
	 * In this case, we do the following.
	 * 1) We find the index of the neighboring face (e.g. for t1 need index of t2).
	 * 2) We check if the neighboring face (t2) is already in the top K faces.
	 * 3) If it is, compare the distance of the pixel to t1 with the distance to t2.
	 * 4) If distance(point,t1) < distance(point,t2), overwrite the values for t2 in the top K faces.
	 */



}

} // namespace nnrt::rendering::kernel