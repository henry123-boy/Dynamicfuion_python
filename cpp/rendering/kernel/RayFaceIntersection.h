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
#include "rendering/functional/kernel/BarycentricCoordinates.h"


namespace o3tgk = open3d::t::geometry::kernel;
namespace nnrt::rendering::kernel {
using t_face_index = int32_t;

struct RayFaceIntersection {
	float depth; // depth of the pixel ray intersection (in normalized camera coordinates)
	t_face_index face_index;
	// signed distance of pixel ray to face in XY plane (in normalized camera coordinates), i.e. distance to the nearest triangle edge
	// negative for "inside triangle", positive for "outside triangle"
	float distance;
	Eigen::Vector3f barycentric_coordinates;
};

NNRT_DEVICE_WHEN_CUDACC
inline
bool operator<(const RayFaceIntersection& a, const RayFaceIntersection& b) {
	return a.depth < b.depth || (a.depth == b.depth && a.face_index < b.face_index);
}


template<typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline void ComputeFace2dBoundingBoxAndCheckZMin(
		float& x_min,
		float& x_max,
		float& y_min,
		float& y_max,
		bool& z_invalid,
		const float blur_radius,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2
) {
	x_min = FloatMin3(vertex0.x(), vertex1.x(), vertex2.x()) - blur_radius;
	x_max = FloatMax3(vertex0.x(), vertex1.x(), vertex2.x()) + blur_radius;

	y_min = FloatMin3(vertex0.y(), vertex1.y(), vertex2.y()) - blur_radius;
	y_max = FloatMax3(vertex0.y(), vertex1.y(), vertex2.y()) + blur_radius;

	const float z_min = FloatMin3(vertex0.z(), vertex1.z(), vertex2.z());

	// Faces with at least one vertex behind the camera won't render correctly
	// and should be removed or clipped before rasterizing
	z_invalid = z_min < K_EPSILON;
}

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
	float x_min, x_max, y_min, y_max;
	bool z_invalid;
	ComputeFace2dBoundingBoxAndCheckZMin(x_min, x_max, y_min, y_max, z_invalid, blur_radius, vertex0, vertex1, vertex2);
	return (point.x() > x_max || point.x() < x_min || point.y() > y_max || point.y() < y_min || z_invalid);
}


// Determine whether the point (px, py) lies outside the face 2D bounding box while accounting for the blur radius, then record the bounding box.
template<typename TVertex>
NNRT_DEVICE_WHEN_CUDACC
inline void CalculateAndStoreFace2dBoundingBox(
		float* bounding_box_data,
		bool* skip_mask_data,
		int64_t i_face,
		const int64_t face_count,
		const TVertex& vertex0,
		const TVertex& vertex1,
		const TVertex& vertex2,
		float blur_radius
) {
	float x_min, x_max, y_min, y_max;
	bool z_invalid;
	ComputeFace2dBoundingBoxAndCheckZMin(x_min, x_max, y_min, y_max, z_invalid, blur_radius, vertex0, vertex1, vertex2);

	bounding_box_data[0 * face_count + i_face] = x_min;
	bounding_box_data[1 * face_count + i_face] = x_max;
	bounding_box_data[2 * face_count + i_face] = y_min;
	bounding_box_data[3 * face_count + i_face] = y_max;
	skip_mask_data[i_face] = z_invalid;
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
		const TPoint& point,
		const TTriangleVertex& vertex0,
		const TTriangleVertex& vertex1,
		const TTriangleVertex& vertex2
) {
	// Compute distance to all three edges and return the minimum.
	const float e01_dist = PointSegmentSquareDistance(point, vertex0, vertex1);
	const float e02_dist = PointSegmentSquareDistance(point, vertex0, vertex2);
	const float e12_dist = PointSegmentSquareDistance(point, vertex1, vertex2);
	const float edge_dist = FloatMin3(e01_dist, e02_dist, e12_dist);
	return edge_dist;
}


/*
 * Assumes normalized camera-space coordinates for face vertices and pixel
 */
template<functional::kernel::FrontFaceVertexOrder TVertexOrder = functional::kernel::CounterClockWise>
NNRT_DEVICE_WHEN_CUDACC
inline void UpdateQueueIfPixelInsideFace(
		const o3tgk::TArrayIndexer <t_face_index>& face_vertex_position_indexer,
		t_face_index i_face,
		RayFaceIntersection* queue,
		int& queue_size,
		float& queue_max_depth,
		int& queue_max_depth_at,
		float blur_radius,
		const Eigen::Vector2f& pixel,
		const int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates,
		bool clip_barycentric_coordinates,
		bool cull_back_faces
) {
	auto face_vertices_data = face_vertex_position_indexer.GetDataPtr<float>(i_face);
	Eigen::Map<Eigen::Vector3f> face_vertex0(face_vertices_data);
	Eigen::Map<Eigen::Vector3f> face_vertex1(face_vertices_data + 3);
	Eigen::Map<Eigen::Vector3f> face_vertex2(face_vertices_data + 6);

	Eigen::Map<Eigen::Vector2f> face_vertex0_xy(face_vertices_data);
	Eigen::Map<Eigen::Vector2f> face_vertex1_xy(face_vertices_data + 3);
	Eigen::Map<Eigen::Vector2f> face_vertex2_xy(face_vertices_data + 6);

	// face_area is computed using the CW convention for front-facing triangles (not the default CCW convention as in OpenGL).
	const float face_area =
			functional::kernel::SignedParallelogramArea<TVertexOrder>(
					face_vertex0_xy, face_vertex1_xy, face_vertex2_xy
			);
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

	Eigen::Vector3f barycentric_coordinates = functional::kernel::BarycentricCoordinates<TVertexOrder>(
			pixel, face_vertex0_xy, face_vertex1_xy, face_vertex2_xy
	);
	if (perspective_correct_barycentric_coordinates) {
		barycentric_coordinates =
				functional::kernel::PerspectiveCorrectBarycentricCoordinates(barycentric_coordinates, face_vertex0.z(), face_vertex1.z(), face_vertex2.z());
	}
	Eigen::Vector3f barycentric_coordinates_clipped =
			clip_barycentric_coordinates ? functional::kernel::ClipBarycentricCoordinates(barycentric_coordinates) : barycentric_coordinates;

	const float intersection_depth =
			barycentric_coordinates_clipped.x() * face_vertex0.z() +
			barycentric_coordinates_clipped.y() * face_vertex1.z() +
			barycentric_coordinates_clipped.z() * face_vertex2.z();
	if (intersection_depth < 0.f) {
		return; // Face is behind image plane.
	}
	const float point_face_distance = PointTriangleDistance(pixel, face_vertex0_xy, face_vertex1_xy, face_vertex2_xy);

	// Use the unclipped barycentric coordinates to determine if the point is inside the face.
	const bool inside = barycentric_coordinates.x() > 0.f && barycentric_coordinates.y() > 0.f && barycentric_coordinates.z() > 0.f;
	const float signed_point_face_distance = inside ? -point_face_distance : point_face_distance;

	// If pixel is both outside and farther away than the blur radius, exit
	if (!inside && point_face_distance >= blur_radius) {
		return;
	}

	if (queue_size < faces_per_pixel) {
		// Add the intersection to the queue
		queue[queue_size] = {intersection_depth, i_face, signed_point_face_distance, barycentric_coordinates_clipped};
		// If intersection is beyond the max intersection depth in the queue, update the maximum depth and its index.
		if (intersection_depth > queue_max_depth) {
			queue_max_depth = intersection_depth;
			queue_max_depth_at = queue_size;
		}
		queue_size++;
	} else if (intersection_depth < queue_max_depth) {
		// overwrite the maximum-depth face intersection info with the current faces' intersection data, find the new maximum depth,
		// and use that to update the maximum depth and its index.
		queue[queue_max_depth_at] = {intersection_depth, i_face, signed_point_face_distance, barycentric_coordinates_clipped};
		queue_max_depth = intersection_depth;
		for (int i_pixel_face = 0; i_pixel_face < faces_per_pixel; i_pixel_face++) {
			if (queue[i_pixel_face].depth > queue_max_depth) {
				queue_max_depth = queue[i_pixel_face].depth;
				queue_max_depth_at = i_pixel_face;
			}
		}
	}
}

} // namespace nnrt::rendering::kernel