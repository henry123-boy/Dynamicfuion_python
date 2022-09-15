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
#include "functional.h"

#include "geometry/functional/WarpAnchorComputation.h"
#include "geometry/functional/Warping.h"

namespace o3tg = open3d::t::geometry;
namespace o3c = open3d::core;

namespace nnrt::geometry::functional {

void pybind_geometry_functional(pybind11::module& m) {
	auto core_module = py::module::import("open3d.core");
	py::module::import("open3d.cuda.pybind.t.geometry");
	auto size_vector_class = core_module.attr("SizeVector");

	py::module m_submodule = m.def_submodule(
			"functional", "Module with functions acting on Open3D geometry objects and Open3D-tensor-based geometry objects."
	);

	pybind_geometry_functional_warp_anchor_computation(m_submodule);
	pybind_geometry_functional_warping(m_submodule);
}

void pybind_geometry_functional_warp_anchor_computation(pybind11::module& m) {
	m.def("compute_anchors_and_weights_euclidean", py::overload_cast<const o3c::Tensor&, const o3c::Tensor&, int, int,
			      float>(&ComputeAnchorsAndWeightsEuclidean), "points"_a, "nodes"_a, "anchor_count"_a,
	      "minimum_valid_anchor_count"_a, "node_coverage"_a);

	m.def("compute_anchors_and_weights_shortest_path", py::overload_cast<const o3c::Tensor&, const o3c::Tensor&,
			      const o3c::Tensor&, int, float>(&ComputeAnchorsAndWeightsShortestPath), "points"_a, "nodes"_a, "edges"_a,
	      "anchor_count"_a, "node_coverage"_a);
}

void pybind_geometry_functional_warping(pybind11::module& m) {
	m.def("warp_triangle_mesh", &WarpTriangleMesh, "input_mesh"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchor_count"_a, "node_coverage"_a, "threshold_nodes_by_distance"_a = false,
	      "minimum_valid_anchor_count"_a = 0, "extrinsics"_a = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")));

	m.def("warp_point_cloud", py::overload_cast<const open3d::t::geometry::PointCloud&, const o3c::Tensor&,
			      const o3c::Tensor&, const o3c::Tensor&, int, float, int, const o3c::Tensor&>(&WarpPointCloud),
	      "input_point_cloud"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchor_count"_a, "node_coverage"_a,
	      "minimum_valid_anchor_count"_a, "extrinsics"_a = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")));

	m.def("warp_point_cloud", py::overload_cast<const open3d::t::geometry::PointCloud&, const o3c::Tensor&,
			      const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&, int, const o3c::Tensor&>(&WarpPointCloud),
	      "input_point_cloud"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchors"_a, "anchor_weights"_a,
	      "minimum_valid_anchor_count"_a, "extrinsics"_a = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")));
}

} // namespace nnrt::geometry::functional