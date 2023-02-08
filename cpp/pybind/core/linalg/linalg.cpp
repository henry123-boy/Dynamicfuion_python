//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/8/23.
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
// stdlib includes

// third-party includes

// local includes
#include "linalg.h"
#include "core/linalg/Rodrigues.h"

namespace nnrt::core::linalg {
void pybind_core_linalg(py::module& m){
	auto core_module = py::module::import("open3d.core");
	py::module m_submodule = m.def_submodule(
			"linalg", "Contains additional linear algebra functions not defined in open3d.core or its submodules."
	);
	m_submodule.def("AxisAngleVectorsToMatricesRodrigues", &core::linalg::AxisAngleVectorsToMatricesRodrigues, "vectors"_a);
}

} //  nnrt::core::linalg