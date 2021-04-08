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
#pragma once

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

namespace rendering {

py::tuple render_mesh_cpu(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int width, int height,
                          const py::array_t<float>& camera_intrinsic_matrix, float depth_scale_factor);

py::tuple render_mesh_gl(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int width, int height,
                         const py::array_t<float>& camera_intrinsic_matrix, float depth_scale_factor);

} // namespace rendering