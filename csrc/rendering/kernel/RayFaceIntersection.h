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

namespace o3tgk = open3d::t::geometry::kernel;
namespace nnrt::rendering::kernel {
using t_face_index = int32_t;

struct RayFaceIntersection {
	float depth;
	int64_t face_index; // index of face
	float distance; // distance of intersection to face
	Eigen::Vector3f barycentric_coordinates;
};


/*
 * Assumes normalized camera-space coordinates for face vertices
 */
NNRT_DEVICE_WHEN_CUDACC
inline void UpdateQueueIfPixelInsideFace(
		const o3tgk::TArrayIndexer<t_face_index>& face_vertex_position_indexer,
		t_face_index i_face,
		RayFaceIntersection* queue, int queue_size,
		float queue_max_depth, int queue_max_depth_at,
		float blur_radius, const Eigen::Vector2f& point_screen,
		const int faces_per_pixel,
		bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces
) {
	auto face_vertices_data = face_vertex_position_indexer.GetDataPtr<float>(i_face);
	Eigen::Map<Eigen::Vector3f> face_vertex0(face_vertices_data);
	Eigen::Map<Eigen::Vector3f> face_vertex1(face_vertices_data + 3);
	Eigen::Map<Eigen::Vector3f> face_vertex2(face_vertices_data + 6);
	//TODO
}

} // namespace nnrt::rendering::kernel