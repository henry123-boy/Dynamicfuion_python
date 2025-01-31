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
#include <open3d/geometry/Image.h>

// nnrt_cpp
#include "geometry/NonRigidSurfaceVoxelBlockGrid.h"
#include "geometry/HierarchicalGraphWarpField.h"
#include "geometry/functional/AnchorComputationMethod.h"
#include "geometry/WarpNodeCoverageComputationMethod.h"
#include "geometry/TransformationMode.h"

#include "geometry/functional/GeometrySampling.h"

// local
#include "pybind/geometry/functional/functional.h"
#include "geometry.h"


#include "pybind/enum_export.h"

namespace o3tg = open3d::t::geometry;
namespace o3c = open3d::core;

namespace nnrt::geometry {
void pybind_geometry(py::module& m) {
	py::module m_submodule = m.def_submodule("geometry", "Module with Open3D-tensor-based geometry objects.");
	functional::pybind_geometry_functional(m_submodule);

	pybind_geometry_enums(m_submodule);
	pybind_geometry_voxel_block_grid(m_submodule);

	pybind_geometry_non_rigid_surface_voxel_block_grid(m_submodule);
	pybind_geometry_graph_warp_field(m_submodule);
}

void pybind_geometry_enums(pybind11::module& m) {
	//Examples for how to export enums to Python
	nnrt::export_enum<nnrt::geometry::AnchorComputationMethod>(m);
	nnrt::export_enum<nnrt::geometry::WarpNodeCoverageComputationMethod>(m);
}


void pybind_geometry_voxel_block_grid(pybind11::module& m) {
	auto core_module = py::module::import("open3d.core");
	py::module::import("open3d.cuda.pybind.t.geometry");
	auto size_vector_class = core_module.attr("SizeVector");

	py::class_<VoxelBlockGrid> voxel_block_grid(
			m, "VoxelBlockGrid",
			"A voxel block grid is a sparse grid of voxel blocks. Each voxel "
			"block is a dense 3D array, preserving local data distribution. If "
			"the block_resolution is set to 1, then the VoxelBlockGrid "
			"degenerates to a sparse voxel grid.");

	voxel_block_grid.def(py::init<>());

	voxel_block_grid.def(py::init<const std::vector<std::string>&,
			                     const std::vector<o3c::Dtype>&,
			                     const std::vector<o3c::SizeVector>&, float, int64_t,
			                     int64_t, const o3c::Device&>(),
	                     "attr_names"_a, "attr_dtypes"_a, "attr_channels"_a,
	                     "voxel_size"_a = 0.0058, "block_resolution"_a = 16,
	                     "block_count"_a = 10000, "device"_a = o3c::Device("CPU:0"));

	voxel_block_grid.def("hashmap", &VoxelBlockGrid::GetHashMap,
	                     "Get the underlying hash map from 3d block coordinates to block "
	                     "voxel grids.");

	voxel_block_grid.def("attribute", &VoxelBlockGrid::GetAttribute,
	                     "Get the attribute tensor to be indexed with voxel_indices.",
	                     "attribute_name"_a);

	voxel_block_grid.def("voxel_indices",
	                     py::overload_cast<const o3c::Tensor&>(
			                     &VoxelBlockGrid::GetVoxelIndices, py::const_),
	                     "Get a (4, N), Int64 index tensor for input buffer indices, used "
	                     "for advanced indexing.   "
	                     "Returned index tensor can access selected value buffer"
	                     "in the order of  "
	                     "(buf_index, index_voxel_x, index_voxel_y, index_voxel_z).       "
	                     "Example:                                                        "
	                     "For a voxel block grid with (2, 2, 2) block resolution,         "
	                     "if the active block coordinates are at buffer index {(2, 4)} "
	                     "given by active_indices() from the underlying hash map,         "
	                     "the returned result will be a (4, 2 x 8) tensor:                "
	                     "{                                                               "
	                     "(2, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 1, 1, 0),         "
	                     "(2, 0, 0, 1), (2, 1, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1),         "
	                     "(4, 0, 0, 0), (4, 1, 0, 0), (4, 0, 1, 0), (4, 1, 1, 0),         "
	                     "(4, 0, 0, 1), (4, 1, 0, 1), (4, 0, 1, 1), (4, 1, 1, 1),         "
	                     "}"
	                     "Note: the slicing order is z-y-x.");

	voxel_block_grid.def("voxel_indices",
	                     py::overload_cast<>(&VoxelBlockGrid::GetVoxelIndices, py::const_),
	                     "Get a (4, N) Int64 idnex tensor for all the active voxels stored "
	                     "in the hash map, used for advanced indexing.");

	voxel_block_grid.def("voxel_coordinates", &VoxelBlockGrid::GetVoxelCoordinates,
	                     "Get a (3, hashmap.Size() * resolution^3) coordinate tensor of "
	                     "active"
	                     "voxels per block, used for geometry transformation jointly with   "
	                     "indices from voxel_indices.                                       "
	                     "Example:                                                          "
	                     "For a voxel block grid with (2, 2, 2) block resolution,           "
	                     "if the active block coordinates are {(-1, 3, 2), (0, 2, 4)},      "
	                     "the returned result will be a (3, 2 x 8) tensor given by:         "
	                     "{                                                                 "
	                     "key_tensor[voxel_indices[0]] * block_resolution_ + "
	                     "voxel_indices[1] "
	                     "key_tensor[voxel_indices[0]] * block_resolution_ + "
	                     "voxel_indices[2] "
	                     "key_tensor[voxel_indices[0]] * block_resolution_ + "
	                     "voxel_indices[3] "
	                     "}                                                                 "
	                     "Note: the coordinates are VOXEL COORDINATES in Int64. To access "
	                     "metric"
	                     "coordinates, multiply by voxel size.",
	                     "voxel_indices"_a);

	voxel_block_grid.def("voxel_coordinates_and_flattened_indices",
	                     py::overload_cast<const o3c::Tensor&>(
			                     &VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices),
	                     "Get a (buf_indices.shape[0] * resolution^3, 3), Float32 voxel "
	                     "coordinate tensor,"
	                     "and a (buf_indices.shape[0] * resolution^3, 1), Int64 voxel index "
	                     "tensor.",
	                     "buf_indices"_a);

	voxel_block_grid.def("voxel_coordinates_and_flattened_indices",
	                     py::overload_cast<>(
			                     &VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices),
	                     "Get a (hashmap.size() * resolution^3, 3), Float32 voxel "
	                     "coordinate tensor,"
	                     "and a (hashmap.size() * resolution^3, 1), Int64 voxel index "
	                     "tensor.");

	voxel_block_grid.def("compute_unique_block_coordinates",
	                     py::overload_cast<const o3tg::Image&, const o3c::Tensor&,
			                     const o3c::Tensor&, float, float, float>(
			                     &VoxelBlockGrid::GetUniqueBlockCoordinates),
	                     "Get a (3, M) active block coordinates from a depth image, with "
	                     "potential duplicates removed."
	                     "Note: these coordinates are not activated in the internal sparse "
	                     "voxel block. They need to be inserted in the hash map.",
	                     "depth"_a, "intrinsic"_a, "extrinsic"_a, "depth_scale"_a = 1000.0f,
	                     "depth_max"_a = 3.0f, "trunc_voxel_multiplier"_a = 8.0);

	voxel_block_grid.def("compute_unique_block_coordinates",
	                     py::overload_cast<const o3tg::PointCloud&, float>(
			                     &VoxelBlockGrid::GetUniqueBlockCoordinates),
	                     "Obtain active block coordinates from a point cloud.", "pcd"_a,
	                     "trunc_voxel_multiplier"_a = 8.0);

	voxel_block_grid.def("integrate",
	                     py::overload_cast<const o3c::Tensor&, const o3tg::Image&, const o3tg::Image&,
			                     const o3c::Tensor&, const o3c::Tensor&,
			                     const o3c::Tensor&, float, float, float>(
			                     &VoxelBlockGrid::Integrate),
	                     "Specific operation for TSDF volumes."
	                     "Integrate an RGB-D frame in the selected block coordinates using "
	                     "pinhole camera model.",
	                     "block_coords"_a, "depth"_a, "color"_a, "depth_intrinsic"_a,
	                     "color_intrinsic"_a, "extrinsic"_a,
	                     "depth_scale"_a.noconvert() = 1000.0f,
	                     "depth_max"_a.noconvert() = 3.0f,
	                     "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

	voxel_block_grid.def("integrate",
	                     py::overload_cast<const o3c::Tensor&, const o3tg::Image&, const o3tg::Image&,
			                     const o3c::Tensor&, const o3c::Tensor&, float,
			                     float, float>(&VoxelBlockGrid::Integrate),
	                     "Specific operation for TSDF volumes."
	                     "Integrate an RGB-D frame in the selected block coordinates using "
	                     "pinhole camera model.",
	                     "block_coords"_a, "depth"_a, "color"_a, "intrinsic"_a,
	                     "extrinsic"_a, "depth_scale"_a.noconvert() = 1000.0f,
	                     "depth_max"_a.noconvert() = 3.0f,
	                     "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

	voxel_block_grid.def("integrate",
	                     py::overload_cast<const o3c::Tensor&, const o3tg::Image&,
			                     const o3c::Tensor&, const o3c::Tensor&, float,
			                     float, float>(&VoxelBlockGrid::Integrate),
	                     "Specific operation for TSDF volumes."
	                     "Similar to RGB-D integration, but only applied to depth images.",
	                     "block_coords"_a, "depth"_a, "intrinsic"_a, "extrinsic"_a,
	                     "depth_scale"_a.noconvert() = 1000.0f,
	                     "depth_max"_a.noconvert() = 3.0f,
	                     "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

	voxel_block_grid.def("ray_cast", &VoxelBlockGrid::RayCast,
	                     "Specific operation for TSDF volumes."
	                     "Perform volumetric ray casting in the selected block coordinates."
	                     "The block coordinates in the frustum can be taken from"
	                     "compute_unique_block_coordinates"
	                     "All the block coordinates can be taken from "
	                     "hashmap().key_tensor()",
	                     "block_coords"_a, "intrinsic"_a, "extrinsic"_a, "width"_a,
	                     "height"_a,
	                     "render_attributes"_a = std::vector<std::string>{"depth", "color"},
	                     "depth_scale"_a = 1000.0f, "depth_min"_a = 0.1f,
	                     "depth_max"_a = 3.0f, "weight_threshold"_a = 3.0f,
	                     "trunc_voxel_multiplier"_a = 8.0f, "range_map_down_factor"_a = 8);

	voxel_block_grid.def("extract_point_cloud", &VoxelBlockGrid::ExtractPointCloud,
	                     "Specific operation for TSDF volumes."
	                     "Extract point cloud at isosurface points.",
	                     "weight_threshold"_a = 3.0f, "estimated_point_number"_a = -1);

	voxel_block_grid.def("extract_triangle_mesh", &VoxelBlockGrid::ExtractTriangleMesh,
	                     "Specific operation for TSDF volumes."
	                     "Extract triangle mesh at isosurface points.",
	                     "weight_threshold"_a = 3.0f, "estimated_vertex_number"_a = -1);

	voxel_block_grid.def("save", &VoxelBlockGrid::Save,
	                     "Save the voxel block grid to a npz file."
	                     "file_name"_a);
	voxel_block_grid.def_static("load", &VoxelBlockGrid::Load,
	                            "Load a voxel block grid from a npz file.", "file_name"_a);

	voxel_block_grid.def("get_device", &VoxelBlockGrid::GetDevice);
	voxel_block_grid.def("get_block_resolution", &VoxelBlockGrid::GetBlockResolution);
	voxel_block_grid.def("get_block_count", &VoxelBlockGrid::GetBlockCount);
}

void pybind_geometry_non_rigid_surface_voxel_block_grid(pybind11::module& m) {

	py::class_<NonRigidSurfaceVoxelBlockGrid, VoxelBlockGrid> non_rigid_surface_voxel_block_grid(
			m, "NonRigidSurfaceVoxelBlockGrid",
			"A voxel block grid is a sparse grid of voxel blocks. Each voxel "
			"block is a dense 3D array, preserving local data distribution. If "
			"the block_resolution is set to 1, then the VoxelBlockGrid "
			"degenerates to a sparse voxel grid.");

	non_rigid_surface_voxel_block_grid.def(py::init<const std::vector<std::string>&,
			                                       const std::vector<o3c::Dtype>&,
			                                       const std::vector<o3c::SizeVector>&, float, int64_t,
			                                       int64_t, const o3c::Device&>(),
	                                       "attr_names"_a, "attr_dtypes"_a, "attr_channels"_a,
	                                       "voxel_size"_a = 0.0058, "block_resolution"_a = 16,
	                                       "block_count"_a = 10000, "device"_a = o3c::Device("CPU:0"));

	non_rigid_surface_voxel_block_grid.def("find_blocks_intersecting_truncation_region",
	                                       &NonRigidSurfaceVoxelBlockGrid::FindBlocksIntersectingTruncationRegion,
	                                       "depth"_a, "warp_field"_a, "intrinsics"_a, "extrinsics"_a, "depth_scale"_a, "depth_max"_a,
	                                       "truncation_voxel_multiplier"_a);

	non_rigid_surface_voxel_block_grid.def("integrate_non_rigid", py::overload_cast<
			                                       const o3c::Tensor&, const WarpField&,
			                                       const o3tg::Image&, const o3tg::Image&, const o3c::Tensor&,
			                                       const o3c::Tensor&, const o3c::Tensor&, const o3c::Tensor&,
			                                       float, float, float
	                                       >(&NonRigidSurfaceVoxelBlockGrid::IntegrateNonRigid),
	                                       "block_coords"_a, "warp_field"_a,
	                                       "depth"_a, "color"_a, "depth_normals"_a,
	                                       "depth_intrinsics"_a, "color_intrinsics"_a, "extrinsics"_a,
	                                       "depth_scale"_a, "depth_max"_a, "truncation_voxel_multiplier"_a);

	non_rigid_surface_voxel_block_grid.def("extract_voxel_values_and_coordinates", &NonRigidSurfaceVoxelBlockGrid::ExtractVoxelValuesAndCoordinates);
	non_rigid_surface_voxel_block_grid.def("extract_voxel_block_coordinates", &NonRigidSurfaceVoxelBlockGrid::ExtractVoxelBlockCoordinates);
	non_rigid_surface_voxel_block_grid.def("activate_sleeve_blocks", &NonRigidSurfaceVoxelBlockGrid::ActivateSleeveBlocks);
}

void pybind_geometry_graph_warp_field(pybind11::module& m) {

	py::class_<WarpField> graph_warp_field(
			m, "GraphWarpField",
			"A warp (motion) field represented by a graph (i.e. using a bidirectional graph as the motion proxy)."
			"Motion of a particular point within the field can be represented by a weighted function of the motions of "
			"the graph nodes within the proximity of this point. Currently, only supports linear & rotational node motion "
			"for a single transformation (not suitable for storing animation data all by itself)."
	);
	graph_warp_field.def(py::init<o3c::Tensor&, float, bool, int, int>(),
	                     "nodes"_a,
	                     "node_coverage"_a = 0.05, "threshold_nodes_by_distance"_a = false,
	                     "anchor_count"_a = 4, "minimum_valid_anchor_count"_a = 0);
	graph_warp_field.def("get_warped_nodes", &WarpField::GetWarpedNodes);
	graph_warp_field.def("get_node_extent", &WarpField::GetNodeExtent);
	graph_warp_field.def("warp_mesh",
	                     py::overload_cast<const open3d::t::geometry::TriangleMesh&, bool,
			                     const open3d::core::Tensor&>(&WarpField::WarpMesh, py::const_),
	                     "input_mesh"_a, "disable_neighbor_thresholding"_a = true,
	                     "extrinsics"_a = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")));
	graph_warp_field.def("warp_mesh",
	                     py::overload_cast<const open3d::t::geometry::TriangleMesh&, const open3d::core::Tensor&, const open3d::core::Tensor&,
			                     bool, const open3d::core::Tensor&>(&WarpField::WarpMesh, py::const_),
	                     "input_mesh"_a, "anchors"_a, "anchor_weights"_a, "disable_neighbor_thresholding"_a = true,
	                     "extrinsics"_a = open3d::core::Tensor::Eye(4, open3d::core::Float64, open3d::core::Device("CPU:0")));
	graph_warp_field.def("clone", &WarpField::Clone);

	graph_warp_field.def("reset_rotations", &WarpField::ResetRotations);
	graph_warp_field.def("apply_transformations", &WarpField::ApplyTransformations);

	graph_warp_field.def("get_node_rotations", py::overload_cast<>(&WarpField::GetNodeRotations));
	graph_warp_field.def("get_node_translations", py::overload_cast<>(&WarpField::GetNodeTranslations));
	graph_warp_field.def("set_node_rotations", &WarpField::SetNodeRotations, "node_rotations"_a);
	graph_warp_field.def("set_node_translations", &WarpField::SetNodeTranslations, "node_translations"_a);
	graph_warp_field.def("translate_nodes", &WarpField::TranslateNodes, "node_translation_deltas"_a);
	graph_warp_field.def("rotate_nodes", &WarpField::RotateNodes, "node_rotation_deltas"_a);

	graph_warp_field.def_readonly("nodes", &WarpField::node_positions);
	//TODO: expose ~_graph_warp_field classes
	// graph_warp_field.def_readonly("edges", &WarpField::edges);
	// graph_warp_field.def_readonly("edge_weights", &WarpField::edge_weights);
	// graph_warp_field.def_readonly("clusters", &WarpField::clusters);
}


} // namespace nnrt

