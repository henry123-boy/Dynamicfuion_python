//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/29/21.
//  Copyright (c) 2021 Gregory Kramida
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
#include "rendering.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

namespace rendering {

static const float kEpsilon = 1e-8;
static const float kInfinity = std::numeric_limits<float>::max();

inline
bool compute_ray_triangle_intersection(
		const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction,
		const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
		float& t, float& u, float& v) {
	Vector3f v0_to_v1 = v1 - v0;
	Vector3f v0_to_v2 = v2 - v0;
	Vector3f pvec = ray_direction.cross(v0_to_v2);
	float det = v0_to_v1.dot(pvec);

	// ray and triangle are parallel if det is close to 0
	if (abs(det) < kEpsilon) return false;

	float determinant_reciprocal = 1 / det;

	Vector3f tvec = ray_origin - v0;
	u = tvec.dot(pvec) * determinant_reciprocal;
	if (u < 0 || u > 1) return false;

	Vector3f qvec = tvec.cross(v0_to_v1);
	v = ray_direction.dot(qvec) * determinant_reciprocal;
	if (v < 0 || u + v > 1) return false;

	t = v0_to_v2.dot(qvec) * determinant_reciprocal;

	return true;
}


// Test if the ray intersects this triangle mesh
inline
bool intersect(const Vector3f& ray_origin, const Vector3f& ray_direction,
               float& t_near,
               const py::array_t<float>& vertex_positions,
               const py::array_t<int>& face_indices,
               int& triangle_index,
               Vector2f& uv) {
	bool intersection_found = false;
	const int face_count = face_indices.shape(0);
	for (int face_idx = 0; face_idx < face_count; ++face_idx) {
		Vector3i face(*face_indices.data(face_idx, 0), *face_indices.data(face_idx, 1), *face_indices.data(face_idx, 2));

		Vector3f v0(*vertex_positions.data(face[0], 0),
		            *vertex_positions.data(face[0], 1),
		            *vertex_positions.data(face[0], 2));
		Vector3f v1(*vertex_positions.data(face[1], 0),
		            *vertex_positions.data(face[1], 1),
		            *vertex_positions.data(face[1], 2));
		Vector3f v2(*vertex_positions.data(face[2], 0),
		            *vertex_positions.data(face[2], 1),
		            *vertex_positions.data(face[2], 2));
		float t = kInfinity, u, v;
		if (compute_ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2, t, u, v) && t < t_near) {
			t_near = t;
			uv.x() = u;
			uv.y() = v;
			triangle_index = face_idx;
			intersection_found |= true;
		}
	}

	return intersection_found;
}

inline
unsigned short cast_ray(
		const Vector3f& ray_origin, const Vector3f& ray_direction,
		const py::array_t<float>& vertex_positions,
		const py::array_t<int>& face_indices,
		Vector3f& hit_point,
		const float depth_scale_factor) {
	unsigned short hit_depth = 0;
	float t_near = kInfinity;
	Vector2f uv;
	int index = 0;
	if (intersect(ray_origin, ray_direction, t_near, vertex_positions, face_indices, index, uv)) {
		hit_point = ray_origin + ray_direction * t_near;
		hit_depth = static_cast<unsigned short>(hit_point.z() * depth_scale_factor);
	}
	return hit_depth;
}

/**
 * \brief Render a depth image and a point image of a mesh by simple raycasting
 * \details assumes all coordinates are give in camera-space
 * \param vertex_positions Nx3 array containing positions of mesh vertices, one vertex per row
 * \param face_indices Mx3 array containing positions indices of vertices used by each (triangular) mesh face
 * \param width with of the image, in pixels
 * \param height height of the image, in pixels
 * \param camera_intrinsic_matrix 3x3 camera intrinsic matrix
 * \param depth_scale_factor a factor to scale the depth values
 * \return a tuple containing the resulting depth image (unsigned short width x height array)
 * and point image (ordered point cloud, a float width x height x 3 array)
 */
py::tuple render_mesh(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, const int width, const int height,
                      const py::array_t<float>& camera_intrinsic_matrix, const float depth_scale_factor) {

	assert(camera_intrinsic_matrix.ndim() == 2);
	assert(camera_intrinsic_matrix.shape(0) == 3 && camera_intrinsic_matrix.shape(1) == 3);
	assert(vertex_positions.ndim() == 2);
	assert(vertex_positions.shape(1) == 3);
	assert(face_indices.ndim() == 2);
	assert(face_indices.shape(1) == 3);

	py::array_t<unsigned short> depth_image({static_cast<ssize_t>(height), static_cast<ssize_t>(width)});
	py::array_t<float> point_image({static_cast<ssize_t>(height), static_cast<ssize_t>(width), static_cast<ssize_t>(3)});

	const float& f_x = *camera_intrinsic_matrix.data(0, 0);
	const float& f_y = *camera_intrinsic_matrix.data(1, 1);
	const float& c_x = *camera_intrinsic_matrix.data(0, 2);
	const float& c_y = *camera_intrinsic_matrix.data(1, 2);

	// Everything is assumed to be in camera space here
	const Vector3f camera_origin(0.f, 0.f, 0.f);

#pragma omp parallel for default(none) shared(depth_image, point_image, vertex_positions, face_indices)\
firstprivate(height, width, f_x, f_y, c_x, c_y, camera_origin, depth_scale_factor)
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			// Generate ray direction
			// assumes pinhole camera (no distortion)
			// X_camera / Z_camera = (u - c_x) / f_x, see pinhole camera
			// equations at https://docs.opencv.org/master/d9/d0c/group__calib3d.html
			// This effectively produces a ray in camera space, in world units (m)
			Vector3f ray_direction((static_cast<float>(u) - c_x) / f_x, (static_cast<float>(v) - c_y) / f_y, 1.0f);
			ray_direction = ray_direction.normalized();
			Vector3f surface_point;
			unsigned short depth = cast_ray(camera_origin, ray_direction, vertex_positions, face_indices,
			                                surface_point, depth_scale_factor);
			*depth_image.mutable_data(v, u) = depth;
			*point_image.mutable_data(v, u, 0) = surface_point.x();
			*point_image.mutable_data(v, u, 1) = surface_point.y();
			*point_image.mutable_data(v, u, 2) = surface_point.z();
		}
	}

	return py::make_tuple(depth_image, point_image);
}


} // namespace rendering