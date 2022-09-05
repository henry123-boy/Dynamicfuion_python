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
#include "rendering/kernel/RasterizeMesh.h"

namespace o3c = open3d::core;

namespace nnrt::rendering::kernel {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeMeshNaive(const open3d::t::geometry::TriangleMesh& mesh, std::tuple<int64_t, int64_t> image_size, float blur_radius, int faces_per_pixel,
                   bool perspective_correct_barycentric_coordinates, bool clip_barycentric_coordinates, bool cull_back_faces) {
	o3c::Device device = mesh.GetDevice();
}

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeMeshFine(const open3d::t::geometry::TriangleMesh& mesh, const o3c::Tensor& bin_faces, std::tuple<int64_t, int64_t> image_size,
                  float blur_radius, int bin_size, int faces_per_pixel, bool perspective_correct_barycentric_coordinates,
				  bool clip_barycentric_coordinates, bool cull_back_faces) {
	o3c::Device device = mesh.GetDevice();
}

open3d::core::Tensor
RasterizeMeshCoarse(const open3d::t::geometry::TriangleMesh& mesh, std::tuple<int64_t, int64_t> image_size, float blur_radius, int bin_size,
                    int max_faces_per_bin) {
	o3c::Device device = mesh.GetDevice();
}


} // namespace nnrt::rendering::kernel