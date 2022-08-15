//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/15/22.
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
// 3rd party
#include "open3d/core/Tensor.h"

//nnrt
#include "geometry/VoxelBlockGrid.h"
#include "io/TensorIO.h"
#include "io/VoxelBlockGridIO.h"
// local
#include "io.h"

namespace o3c = open3d::core;

namespace nnrt::io {

void pybind_io(pybind11::module& m) {
	pybind_tensor_io(m);
	pybind_voxel_block_grid_io(m);
}

void pybind_tensor_io(pybind11::module& m) {
	auto open3d_core_module = py::module::import("open3d.core");
	auto tensor_class = open3d_core_module.attr("Tensor");

	m.def("write_tensor", [](const std::string& path, const open3d::core::Tensor& tensor, bool compressed) {
		      pybind11::gil_scoped_release release;
		      WriteTensor(path, tensor, compressed);
	      }, "Write tensor in binary form.",
	      "path"_a, "tensor"_a, "compressed"_a = true);

	m.def("read_tensor", [](const std::string& path, bool compressed) {
		      pybind11::gil_scoped_release release;
		      return ReadTensor(path, compressed);
	      }, "Read tensor from binary form.",
	      "path"_a, "compressed"_a = true);
}

void pybind_voxel_block_grid_io(pybind11::module& m) {
	auto geometry_module = py::module::import("nnrt.geometry");
	auto voxel_block_grid_class = geometry_module.attr("VoxelBlockGrid");
	m.def("write_voxel_block_grid", [](const std::string& path, const nnrt::geometry::VoxelBlockGrid& voxel_block_grid, bool compressed) {
		      pybind11::gil_scoped_release release;
		      WriteVoxelBlockGrid(path, voxel_block_grid, compressed);
	      }, "Write voxel block hash in binary form.",
	      "path"_a, "voxel_block_grid"_a, "compressed"_a = true);
	m.def("read_voxel_block_grid", [](const std::string& path, bool compressed) {
		      pybind11::gil_scoped_release release;
		      return ReadVoxelBlockGrid(path, compressed);
	      }, "Write voxel block hash in binary form.",
	      "path"_a, "compressed"_a = true);
}


} // namespace nnrt::io