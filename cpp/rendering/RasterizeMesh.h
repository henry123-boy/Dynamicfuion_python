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

#include <cfloat>
#include <tuple>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/TriangleMesh.h>

namespace nnrt::rendering {

open3d::core::Tensor MeshFaceVerticesToNdcSpace(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor> MeshFaceVerticesAndClipMaskToNdcSpace(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor> MeshFaceVerticesAndNormalsAndClipMaskToNdc(
		const open3d::t::geometry::TriangleMesh& camera_space_mesh,
		const open3d::core::Tensor& intrinsic_matrix,
		const open3d::core::SizeVector& image_size,
		float near_clipping_distance = 0.0,
		float far_clipping_distance = INFINITY
);


std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeMesh(
		const open3d::core::Tensor& normalized_camera_space_face_vertices,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> clipped_faces_mask,
		const open3d::core::SizeVector& image_size,
		float blur_radius = 0.f,
		int faces_per_pixel = 8,
		int bin_size = -1,
		int max_faces_per_bin = -1,
		bool perspective_correct_barycentric_coordinates = false,
		bool clip_barycentric_coordinates = false,
		bool cull_back_faces = true
);


} // namespace nnrt::rendering