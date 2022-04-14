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
#include "pybind/core.h"
#include "core/TensorManipulationRoutines.h"
namespace nnrt::core{
void pybind_core(py::module& m) {
	py::module m_submodule = m.def_submodule(
			"core", "Open3D-tensor-based core module.");

	pybind_tensor_routines(m_submodule);
}

void pybind_tensor_routines(pybind11::module& m) {
	m.def("matmul3d", &core::Matmul3D, "array_of_matrices_a"_a, "array_of_matrices_b"_a);
}

} // nnrt::core