//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/15/22.
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
#include "pybind/nnrt_pybind.h"

namespace nnrt::geometry::functional {

void pybind_geometry_functional(py::module& m);

void pybind_geometry_functional_warp_anchor_computation(py::module& m);
void pybind_geometry_functional_warping(py::module& m);
void pybind_geometry_functional_normals_operations(py::module& m);

void pybind_geometry_functional_comparison(py::module& m);
void pybind_geometry_functional_pointcloud(py::module& m);
void pybind_geometry_downsampling(py::module& m);

} // namespace nnrt::geometry::functional