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
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "rendering/functional/ExtractFaceVertices.h"

// local includes
#include "pybind/rendering/functional/functional.h"

namespace nnrt::rendering::functional {

void pybind_rendering_functional(py::module& m) {
	py::module m_submodule = m.def_submodule(
			"functional", "Module with stateless functions for rasterizing or rendering Open3D/NNRT geometry objects."
	);
	pybind_rendering_functional_extract_face_vertices(m_submodule);
	pybind_rendering_functional_interpolate_vertex_attributes(m_submodule);


}

void pybind_rendering_functional_extract_face_vertices(pybind11::module& m) {
	m.def("get_mesh_ndc_face_vertices_and_clip_mask",
	      py::overload_cast<const std::vector<open3d::t::geometry::TriangleMesh>&,
			      const open3d::core::Tensor&,
			      const open3d::core::SizeVector&,
			      float, float
	      >(&nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask),
		  "camera_space_meshes"_a, "intrinsic_matrix"_a, "image_size"_a, "near_clipping_distance"_a = 0.f,
		  "far_clipping_distance"_a = INFINITY
	);

	m.def("get_mesh_ndc_face_vertices_and_clip_mask",
	      py::overload_cast<const open3d::t::geometry::TriangleMesh&,
			      const open3d::core::Tensor&,
			      const open3d::core::SizeVector&,
			      float, float
	      >(&nnrt::rendering::functional::GetMeshNdcFaceVerticesAndClipMask),
	      "camera_space_mesh"_a, "intrinsic_matrix"_a, "image_size"_a, "near_clipping_distance"_a = 0.f,
	      "far_clipping_distance"_a = INFINITY
	);
}

void pybind_rendering_functional_interpolate_vertex_attributes(pybind11::module& m) {
	m.def("interpolate_vertex_attributes", &nnrt::rendering::functional::InterpolateVertexAttributes,
		  "pixel_face_indices", "barycentric_coordinates", "face_vertex_attributes"
		  );
}


} // namespace nnrt::geometry::functional
