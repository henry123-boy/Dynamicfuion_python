//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/27/23.
//  Copyright (c) 2023 Gregory Kramida
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
// nnrt_cpp
#include "rendering/RasterizeNdcTriangles.h"

// local includes
#include "pybind/rendering/rendering.h"
#include "pybind/rendering/functional/functional.h"

namespace o3tg = open3d::t::geometry;
namespace o3c = open3d::core;

namespace nnrt::rendering {

void pybind_rendering(py::module& m) {
	py::module m_submodule = m.def_submodule("rendering", "Module with rendering and rasterization models / classes that use Open3D geometry.");
	functional::pybind_rendering_functional(m_submodule);

	pybind_rasterization_functions(m_submodule);
}

void pybind_rasterization_functions(py::module& m) {
	m.def("rasterize_ndc_triangles", &rendering::RasterizeNdcTriangles,
	      "ndc_face_vertices"_a, "clipped_faces_mask"_a, "image_size"_a, "blur_radius_pixels"_a = 0.f,
	      "faces_per_pixel"_a = 8, "bin_size"_a = -1, "max_faces_per_bin"_a = -1,
	      "perspective_correct_barycentric_coordinates"_a = false, "clip_barycentric_coordinates"_a = false,
	      "cull_back_faces"_a = true
	);
}

} // namespace nnrt::rendering
