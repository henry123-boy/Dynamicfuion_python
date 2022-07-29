
// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/HashMap.h>
#include <open3d/t/geometry/Geometry.h>
#include <open3d/t/geometry/Image.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/geometry/TensorMap.h>
#include <open3d/t/geometry/TriangleMesh.h>


namespace nnrt::geometry {

/// A voxel block grid is a sparse grid of voxel blocks.
/// Each voxel block is a dense 3D array, preserving local data distribution.
/// If the block_resolution is set to 1, then the VoxelBlockGrid degenerates to
/// a sparse voxel grid.
class VoxelBlockGrid {
public:
	VoxelBlockGrid() = default;

	/// \brief Default Constructor.
	/// Example:
	/// VoxelBlockGrid({"tsdf", "weight", "color"},
	///                {open3d::core::Float32, open3d::core::UInt16, open3d::core::UInt16},
	///                {{1}, {1}, {3}},
	///                0.005,
	///                16,
	///                10000,
	///                open3d::core::Device("CUDA:0"),
	///                open3d::core::HashBackendType::Default);
	VoxelBlockGrid(const std::vector<std::string>& attr_names,
	               const std::vector<open3d::core::Dtype>& attr_dtypes,
	               const std::vector<open3d::core::SizeVector>& attr_channels,
	               float voxel_size = 0.0058,
	               int64_t block_resolution = 16,
	               int64_t block_count = 10000,
	               const open3d::core::Device& device = open3d::core::Device("CPU:0"),
	               const open3d::core::HashBackendType& backend =
	               open3d::core::HashBackendType::Default);

	/// Get the underlying hash map that stores values in structure of arrays
	/// (SoA).
	open3d::core::HashMap GetHashMap() { return *block_hashmap_; }

	/// Get the attribute tensor corresponding to the attribute name.
	/// A sugar for hashmap.GetValueTensor(i)
	open3d::core::Tensor GetAttribute(const std::string& attr_name) const;

	/// Get a (4, N), Int64 index tensor for active voxels, used for advanced
	/// indexing.
	/// Returned index tensor can access selected value buffers in order of
	/// (buf_index, index_voxel_x, index_voxel_y, index_voxel_z).
	///
	/// Example:
	/// For a voxel block grid with (2, 2, 2) block resolution,
	/// if the active block coordinates are at buffer index {(2, 4)} given by
	/// GetActiveIndices() from the underlying hash map,
	/// the returned result will be a (4, 2 x 8) tensor:
	/// {
	/// (2, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 1, 1, 0),
	/// (2, 0, 0, 1), (2, 1, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1),
	/// (4, 0, 0, 0), (4, 1, 0, 0), (4, 0, 1, 0), (4, 1, 1, 0),
	/// (4, 0, 0, 1), (4, 1, 0, 1), (4, 0, 1, 1), (4, 1, 1, 1),
	/// }
	/// Note: the slicing order is z-y-x.
	open3d::core::Tensor GetVoxelIndices(const open3d::core::Tensor& buf_indices) const;

	/// Get all active voxel indices.
	open3d::core::Tensor GetVoxelIndices() const;

	/// Get a (3, hashmap.Size() * resolution^3) coordinate tensor of active
	/// voxels per block, used for geometry transformation jointly with
	/// indices from GetVoxelIndices.
	///
	/// Example:
	/// For a voxel block grid with (2, 2, 2) block resolution,
	/// if the active block coordinates are {(-1, 3, 2), (0, 2, 4)},
	/// the returned result will be a (3, 2 x 8) tensor given by:
	/// {
	/// key_tensor[voxel_indices[0]] * block_resolution_ + voxel_indices[1]
	/// key_tensor[voxel_indices[0]] * block_resolution_ + voxel_indices[2]
	/// key_tensor[voxel_indices[0]] * block_resolution_ + voxel_indices[3]
	/// }
	/// Note: the coordinates are VOXEL COORDINATES in Int64. To access metric
	/// coordinates, multiply by voxel size.
	open3d::core::Tensor GetVoxelCoordinates(const open3d::core::Tensor& voxel_indices) const;

	/// Accelerated combination of GetVoxelIndices and GetVoxelCoordinates.
	/// Returns a (N, 3) coordinate in float, and a (N, ) flattened index
	/// tensor, where N is the number of active voxels located at buf_indices.
	std::pair<open3d::core::Tensor, open3d::core::Tensor>
	GetVoxelCoordinatesAndFlattenedIndices(const open3d::core::Tensor& buf_indices);

	/// Same as above, but N is the number of all the active voxels with blocks
	/// stored in the hash map.
	std::pair<open3d::core::Tensor, open3d::core::Tensor>
	GetVoxelCoordinatesAndFlattenedIndices();

	/// Get a (3, M) active block coordinates from a depth image, with potential
	/// duplicates removed.
	/// Note: these coordinates are not activated in the internal sparse voxel
	/// block. They need to be inserted in the hash map.
	open3d::core::Tensor GetUniqueBlockCoordinates(const open3d::t::geometry::Image& depth,
	                                               const open3d::core::Tensor& intrinsic,
	                                               const open3d::core::Tensor& extrinsic,
	                                               float depth_scale = 1000.0f,
	                                               float depth_max = 3.0f,
	                                               float trunc_voxel_multiplier = 8.0);

	/// Obtain active block coordinates from a point cloud.
	open3d::core::Tensor GetUniqueBlockCoordinates(const open3d::t::geometry::PointCloud& pcd,
	                                               float trunc_voxel_multiplier = 8.0);

	/// Specific operation for TSDF volumes.
	/// Integrate an RGB-D frame in the selected block coordinates using pinhole
	/// camera model.
	/// For built-in kernels, we support efficient hash map types for SLAM:
	/// tsdf: float, weight: uint16_t, color: uint16_t
	/// and accurate mode for differentiable rendering:
	/// tsdf/weight/color: float
	/// We assume input data are either raw:
	/// depth: uint16_t, color: uint8_t
	/// or depth/color: float.
	/// To support other types and properties, users should combine
	/// GetUniqueBlockCoordinates, GetVoxelIndices, and GetVoxelCoordinates,
	/// with self-defined operations.
	void Integrate(const open3d::core::Tensor& block_coords,
	               const open3d::t::geometry::Image& depth,
	               const open3d::t::geometry::Image& color,
	               const open3d::core::Tensor& depth_intrinsic,
	               const open3d::core::Tensor& color_intrinsic,
	               const open3d::core::Tensor& extrinsic,
	               float depth_scale = 1000.0f,
	               float depth_max = 3.0f,
	               float trunc_voxel_multiplier = 8.0f);

	/// Specific operation for TSDF volumes.
	/// Similar to RGB-D integration, but uses the same intrinsics for depth and
	/// color.
	void Integrate(const open3d::core::Tensor& block_coords,
	               const open3d::t::geometry::Image& depth,
	               const open3d::t::geometry::Image& color,
	               const open3d::core::Tensor& intrinsic,
	               const open3d::core::Tensor& extrinsic,
	               float depth_scale = 1000.0f,
	               float depth_max = 3.0f,
	               float trunc_voxel_multiplier = 8.0f);

	/// Specific operation for TSDF volumes.
	/// Similar to RGB-D integration, but only applied to depth.
	void Integrate(const open3d::core::Tensor& block_coords,
	               const open3d::t::geometry::Image& depth,
	               const open3d::core::Tensor& intrinsic,
	               const open3d::core::Tensor& extrinsic,
	               float depth_scale = 1000.0f,
	               float depth_max = 3.0f,
	               float trunc_voxel_multiplier = 8.0f);

	/// Specific operation for TSDF volumes.
	/// Perform volumetric ray casting in the selected block coordinates.
	/// Return selected properties from the frame.
	/// Supported attributes:
	/// Conventional rendering: vertex, depth, color, normal, range
	/// Differentiable rendering (voxel-wise): mask, index, (interpolation)
	/// ratio.
	/// The block coordinates in the frustum can be taken from
	/// GetUniqueBlockCoordinates.
	/// All the block coordinates can be taken from GetHashMap().GetKeyTensor().
	open3d::t::geometry::TensorMap RayCast(const open3d::core::Tensor& block_coords,
	                                       const open3d::core::Tensor& intrinsic,
	                                       const open3d::core::Tensor& extrinsic,
	                                       int width,
	                                       int height,
	                                       const std::vector<std::string> attrs = {"depth", "color"},
	                                       float depth_scale = 1000.0f,
	                                       float depth_min = 0.1f,
	                                       float depth_max = 3.0f,
	                                       float weight_threshold = 3.0f,
	                                       float trunc_voxel_multiplier = 8.0f,
	                                       int range_map_down_factor = 8);

	/// Specific operation for TSDF volumes.
	/// Extract point cloud at isosurface points.
	/// Weight threshold is used to filter outliers. By default we use 3.0,
	/// where we assume a reliable surface point comes from the fusion of at
	/// least 3 viewpoints. Use as low as 0.0 to accept all the possible
	/// observations.
	/// Estimated point numbers optionally speeds up the process by a one-pass
	/// extraction with pre-allocated buffers. Use -1 when no estimate is
	/// available.
	open3d::t::geometry::PointCloud ExtractPointCloud(float weight_threshold = 3.0f,
	                                                  int estimated_point_number = -1);

	/// Specific operation for TSDF volumes.
	/// Extract mesh near iso-surfaces with Marching Cubes.
	/// Weight threshold is used to filter outliers. By default we use 3.0,
	/// where we assume a reliable surface point comes from the fusion of at
	/// least 3 viewpoints. Use as low as 0.0 to accept all the possible
	/// observations.
	/// Estimated point numbers optionally speeds up the process by a one-pass
	/// extraction with pre-allocated buffers. Use -1 when no estimate is
	/// available.
	open3d::t::geometry::TriangleMesh ExtractTriangleMesh(float weight_threshold = 3.0f,
	                                                      int estimated_vertex_numer = -1);

	/// Save a voxel block grid to a .npz file.
	void Save(const std::string& file_name) const;

	/// Load a voxel block grid from a .npz file.
	static VoxelBlockGrid Load(const std::string& file_name);
//============= EXTRA CONVENIENCE METHODS ========================

	/// Retrieve the device used for data storage
	open3d::core::Device GetDevice() const;

	/// Retrieve voxel size
	float GetVoxelSize() const;

	/// Retrieve block resolution
	int64_t GetBlockResolution() const;

	/// Retrieve total block count (incl. unactivated blocks)
	int64_t GetBlockCount() const;

	static open3d::t::geometry::TensorMap ConstructTensorMap(const open3d::core::HashMap& block_hashmap,
	                                                         std::unordered_map<std::string, int> name_attr_map);

protected:
	void AssertInitialized() const;

	float voxel_size_ = -1;
	int64_t block_resolution_ = -1;

	// Global hash map: 3D coords -> voxel blocks in SoA.
	std::shared_ptr<open3d::core::HashMap> block_hashmap_;

	// Local hash map: 3D coords -> indices in block_hashmap_.
	std::shared_ptr<open3d::core::HashMap> frustum_hashmap_;

	// Map: attribute name -> index to access the attribute in SoA.
	std::unordered_map<std::string, int> name_attr_map_;

};
}  // namespace nnrt::geometry

