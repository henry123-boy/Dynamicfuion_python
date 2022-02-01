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
#include <open3d/t/geometry/TSDFVoxelGrid.h>
#include <open3d/geometry/Image.h>

#include "geometry/WarpableTSDFVoxelGrid.h"
#include "geometry/GraphWarpField.h"
#include "geometry/AnchorComputationMethod.h"
#include "geometry/TransformationMode.h"
#include "geometry/Comparison.h"
#include "geometry.h"


#include "pybind/enum_export.h"

namespace o3tg = open3d::t::geometry;
namespace o3c = open3d::core;

namespace nnrt::geometry {
void pybind_geometry(py::module& m) {
	py::module m_submodule = m.def_submodule(
			"geometry", "Open3D-tensor-based geometry defining module.");

	pybind_geometry_enums(m_submodule);
	pybind_geometry_extended_tsdf_voxelgrid(m_submodule);
	pybind_geometry_graph_warp_field(m_submodule);
	pybind_geometry_comparison(m_submodule);

}

void pybind_geometry_enums(pybind11::module& m) {
	//Example for how to export enums to Python
	nnrt::export_enum<nnrt::geometry::AnchorComputationMethod>(m);
}


// #define USE_BASE_CLASS_TEMPLATE_PARAMETER

void pybind_geometry_extended_tsdf_voxelgrid(pybind11::module& m) {
#ifdef USE_BASE_CLASS_TEMPLATE_PARAMETER
	// import has to be here in order to load the base class into python,
	// otherwise there will be errors if the module isn't imported on the python end
	py::module_::import("open3d.cuda.pybind.t.geometry");
	py::class_<WarpableTSDFVoxelGrid, open3d::t::geometry::TSDFVoxelGrid> warpable_tsdf_voxel_grid(
			m, "WarpableTSDFVoxelGrid",
			"A voxel grid for TSDF and/or color integration, extended with custom functions for applying a warp-field during integration.");
#else
	py::object tsdf_voxel_grid = (py::object) py::module_::import("open3d.cuda.pybind.t.geometry").attr("TSDFVoxelGrid");
	py::class_<WarpableTSDFVoxelGrid> warpable_tsdf_voxel_grid(
			m, "WarpableTSDFVoxelGrid", tsdf_voxel_grid,
			"A voxel grid for TSDF and/or color integration, extended with custom functions for applying a warp-field during integration.");
#endif
	// TODO: we have to re-define all the TSDFVoxelGrid python aliases, because Open3D code doesn't use the
	//  PYBIND11_EXPORT macro in the class definition. An issue should be raised in Open3D and a pull request proposed to
	//  introduce PYBIND11_EXPORT to various classes and mitigate this problem.
	// region  =============================== EXISTING BASE CLASS (TSDFVoxelGrid) CONSTRUCTORS  ==========================
	warpable_tsdf_voxel_grid.def(
			py::init<const std::unordered_map<std::string, o3c::Dtype>&, float,
					float, int64_t, int64_t, const o3c::Device&>(),
			"map_attrs_to_dtypes"_a =
					std::unordered_map<std::string, o3c::Dtype>{
							{"tsdf",   o3c::Dtype::Float32},
							{"weight", o3c::Dtype::UInt16},
							{"color",  o3c::Dtype::UInt16},
					},
			"voxel_size"_a = 3.0 / 512, "sdf_trunc"_a = 0.04,
			"block_resolution"_a = 16, "block_count"_a = 100,
			"device"_a = o3c::Device("CPU:0"));

	// endregion
	// region  =============================== EXISTING BASE CLASS (TSDFVoxelGrid) FUNCTIONS  ==========================
	//TODO: figure out why do we still need to re-expose these via PyBind11
	warpable_tsdf_voxel_grid.def("integrate",
	                             py::overload_cast<const o3tg::Image&, const o3c::Tensor&,
			                             const o3c::Tensor&, float, float>(
			                             &o3tg::TSDFVoxelGrid::Integrate),
	                             "depth"_a, "intrinsics"_a, "extrinsics"_a,
	                             "depth_scale"_a, "depth_max"_a);
	warpable_tsdf_voxel_grid.def(
			"integrate",
			py::overload_cast<const o3tg::Image&, const o3tg::Image&, const o3c::Tensor&,
					const o3c::Tensor&, float, float>(
					&o3tg::TSDFVoxelGrid::Integrate),
			"depth"_a, "color"_a, "intrinsics"_a, "extrinsics"_a,
			"depth_scale"_a, "depth_max"_a);

	warpable_tsdf_voxel_grid.def(
			"raycast", &o3tg::TSDFVoxelGrid::RayCast, "intrinsics"_a, "extrinsics"_a,
			"width"_a, "height"_a, "depth_scale"_a = 1000.0,
			"depth_min"_a = 0.1f, "depth_max"_a = 3.0f,
			"weight_threshold"_a = 3.0f,
			"raycast_result_mask"_a = o3tg::TSDFVoxelGrid::SurfaceMaskCode::DepthMap |
			                          o3tg::TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
	warpable_tsdf_voxel_grid.def(
			"extract_surface_points", &o3tg::TSDFVoxelGrid::ExtractSurfacePoints,
			"estimate_number"_a = -1, "weight_threshold"_a = 3.0f,
			"surface_mask"_a = o3tg::TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
			                   o3tg::TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
	warpable_tsdf_voxel_grid.def(
			"extract_surface_mesh", &o3tg::TSDFVoxelGrid::ExtractSurfaceMesh,
			"estimate_number"_a = -1, "weight_threshold"_a = 3.0f,
			"surface_mask"_a = o3tg::TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
			                   o3tg::TSDFVoxelGrid::SurfaceMaskCode::ColorMap |
			                   o3tg::TSDFVoxelGrid::SurfaceMaskCode::NormalMap);

	warpable_tsdf_voxel_grid.def("to", &o3tg::TSDFVoxelGrid::To, "device"_a, "copy"_a = false);
	warpable_tsdf_voxel_grid.def("clone", &o3tg::TSDFVoxelGrid::Clone);
	warpable_tsdf_voxel_grid.def(
			"cpu",
			[](const WarpableTSDFVoxelGrid& tsdf_voxelgrid) {
				return tsdf_voxelgrid.To(o3c::Device("CPU:0"));
			},
			"Transfer the tsdf voxelgrid to CPU. If the tsdf voxelgrid "
			"is already on CPU, no copy will be performed.");
	warpable_tsdf_voxel_grid.def(
			"cuda",
			[](const WarpableTSDFVoxelGrid& tsdf_voxelgrid, int device_id) {
				return tsdf_voxelgrid.To(o3c::Device("CUDA", device_id));
			},
			"Transfer the tsdf voxelgrid to a CUDA device. If the tsdf "
			"voxelgrid is already on the specified CUDA device, no copy will "
			"be performed.",
			"device_id"_a = 0);

	warpable_tsdf_voxel_grid.def("get_block_hashmap", [](const WarpableTSDFVoxelGrid& voxelgrid) {
		// Returning shared_ptr can result in double-free.
		return *voxelgrid.GetBlockHashMap();
	});
	warpable_tsdf_voxel_grid.def("get_device", &o3tg::TSDFVoxelGrid::GetDevice);
	// endregion
	// region =============================== EXPOSE CUSTOM / NEW FUNCTIONS =======================================================

	warpable_tsdf_voxel_grid.def("extract_voxel_centers", &WarpableTSDFVoxelGrid::ExtractVoxelCenters);
	warpable_tsdf_voxel_grid.def("extract_tsdf_values_and_weights", &WarpableTSDFVoxelGrid::ExtractTSDFValuesAndWeights);
	warpable_tsdf_voxel_grid.def("extract_values_in_extent", &WarpableTSDFVoxelGrid::ExtractValuesInExtent,
	                             "min_x"_a, "min_y"_a, "min_z"_a,
	                             "max_x"_a, "max_y"_a, "max_z"_a);

	warpable_tsdf_voxel_grid.def("integrate_warped", py::overload_cast<
			                             const o3tg::Image&, const o3tg::Image&, const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&,
										 const GraphWarpField&,
										 float, float
	                             >(&WarpableTSDFVoxelGrid::IntegrateWarped),
	                             "depth"_a, "color"_a, "depth_normals"_a, "intrinsics"_a, "extrinsics"_a,
								 "warp_field"_a, "depth_scale"_a, "depth_max"_a);

	warpable_tsdf_voxel_grid.def("activate_sleeve_blocks", &WarpableTSDFVoxelGrid::ActivateSleeveBlocks);
	// endregion
}

void pybind_geometry_graph_warp_field(pybind11::module& m) {
	m.def("compute_anchors_and_weights_euclidean", py::overload_cast<const o3c::Tensor&, const o3c::Tensor&, int, int,
			      float>(&ComputeAnchorsAndWeightsEuclidean), "points"_a, "nodes"_a, "anchor_count"_a,
	      "minimum_valid_anchor_count"_a, "node_coverage"_a);

	m.def("compute_anchors_and_weights_shortest_path", py::overload_cast<const o3c::Tensor&, const o3c::Tensor&,
			      const o3c::Tensor&, int, float>(&ComputeAnchorsAndWeightsShortestPath), "points"_a, "nodes"_a, "edges"_a,
	      "anchor_count"_a, "node_coverage"_a);

	m.def("warp_triangle_mesh", &WarpTriangleMesh, "input_mesh"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchor_count"_a, "node_coverage"_a, "threshold_nodes_by_distance"_a = false,
	      "minimum_valid_anchor_count"_a = 0);

	m.def("warp_point_cloud", py::overload_cast<const open3d::t::geometry::PointCloud&, const o3c::Tensor&,
			      const o3c::Tensor&, const o3c::Tensor&, int, float, int>(&WarpPointCloud),
	      "input_point_cloud"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchor_count"_a, "node_coverage"_a,
	      "minimum_valid_anchor_count"_a);

	m.def("warp_point_cloud", py::overload_cast<const open3d::t::geometry::PointCloud&, const o3c::Tensor&,
			      const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&, int>(
			      &WarpPointCloud),
	      "input_point_cloud"_a, "nodes"_a, "node_rotations"_a,
	      "node_translations"_a, "anchors"_a, "anchor_weights"_a,
	      "minimum_valid_anchor_count"_a);

	py::class_<GraphWarpField> graph_warp_field(
			m, "GraphWarpField",
			"A warp (motion) field represented by a graph (i.e. using a bidirectional graph as the motion proxy)."
			"Motion of a particular point within the field can be represented by a weighted function of the motions of "
			"the graph nodes within the proximity of this point. Currently, only supports linear & rotational node motion "
			"for a single transformation (not suitable for storing animation data all by itself)."
	);
	graph_warp_field.def(py::init<o3c::Tensor&, o3c::Tensor&, o3c::Tensor&, o3c::Tensor&, float, bool, int, int>(),
	                     "nodes"_a, "edges"_a, "edge_weights"_a, "clusters"_a,
	                     "node_coverage"_a = 0.05, "threshold_nodes_by_distance"_a = false,
	                     "anchor_count"_a = 4, "minimum_valid_anchor_count"_a = 0);
	graph_warp_field.def("get_warped_nodes", &GraphWarpField::GetWarpedNodes);
	graph_warp_field.def("get_node_extent", &GraphWarpField::GetNodeExtent);
	graph_warp_field.def("warp_mesh", &GraphWarpField::WarpMesh,
	                     "input_mesh"_a, "disable_neighbor_thresholding"_a = true);
	graph_warp_field.def_readwrite("nodes", &GraphWarpField::nodes);
	graph_warp_field.def_readwrite("edges", &GraphWarpField::edges);
	graph_warp_field.def_readwrite("edge_weights", &GraphWarpField::edge_weights);
	graph_warp_field.def_readwrite("clusters", &GraphWarpField::clusters);
	graph_warp_field.def_readwrite("translations", &GraphWarpField::translations);
	graph_warp_field.def_readwrite("rotations", &GraphWarpField::rotations);
}

void pybind_geometry_comparison(pybind11::module& m) {
	m.def("compute_point_to_plane_distances",
	      py::overload_cast<const open3d::t::geometry::TriangleMesh&, const open3d::t::geometry::TriangleMesh&>
			      (&ComputePointToPlaneDistances), "mesh1"_a, "mesh2"_a);
	m.def("compute_point_to_plane_distances",
	      py::overload_cast<const open3d::t::geometry::TriangleMesh&, const open3d::t::geometry::PointCloud&>
			      (&ComputePointToPlaneDistances), "mesh"_a, "point_cloud"_a);
}

} // namespace nnrt
