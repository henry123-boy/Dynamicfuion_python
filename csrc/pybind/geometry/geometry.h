//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/6/21.
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
#include "pybind/nnrt_pybind.h"


namespace nnrt::geometry {

void pybind_geometry(py::module& m);
void pybind_geometry_enums(py::module& m);
void pybind_geometry_voxel_block_grid(py::module& m);
void pybind_geometry_non_rigid_surface_voxel_block_grid(py::module& m);
void pybind_geometry_graph_warp_field(py::module& m);
void pybind_geometry_comparison(py::module& m);
void pybind_geometry_downsampling(py::module& m);
void pybind_geometry_pointcloud(py::module& m);
void pybind_geometry_normals_operations(py::module& m);


} //namespace nnrt::geometry



