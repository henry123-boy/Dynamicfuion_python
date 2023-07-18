//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/14/22.
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
#include "pybind/core/core.h"
#include "core/TensorManipulationRoutines.h"
#include "core/KdTree.h"
#include "pybind/core/linalg/linalg.h"


namespace nnrt::core {
void pybind_core(py::module& m) {
	py::module m_submodule = m.def_submodule(
			"core", "Open3D-tensor-based core module.");
	linalg::pybind_core_linalg(m_submodule);
	pybind_tensor_routines(m_submodule);
	pybind_geometry_kd_tree(m_submodule);
}

void pybind_tensor_routines(pybind11::module& m) {
	m.def("matmul3d", &core::Matmul3D, "array_of_matrices_a"_a, "array_of_matrices_b"_a);
}


void pybind_geometry_kd_tree(pybind11::module& m) {
	py::class_<KdTree> kd_tree(m, "KDTree");
	kd_tree.def(py::init<const open3d::core::Tensor&>(), "points"_a);
	kd_tree.def("find_k_nearest_to_points",
				[](KdTree& kd_tree, const open3d::core::Tensor& query_points, int32_t k, bool sort_output = false) {
					open3d::core::Tensor nearest_neighbor_indices;
					open3d::core::Tensor distances;
					kd_tree.FindKNearestToPoints(nearest_neighbor_indices, distances, query_points, k, sort_output);
					return std::make_tuple(nearest_neighbor_indices, distances);
				}, "query_points"_a,  "k"_a, "sort_output"_a = false
	);
}

} // nnrt::core