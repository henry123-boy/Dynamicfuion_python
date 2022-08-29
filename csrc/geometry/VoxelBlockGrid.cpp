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
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/Geometry.h>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/geometry/Utility.h>
#include <open3d/t/geometry/kernel/VoxelBlockGrid.h>
#include <open3d/t/io/NumpyIO.h>
#include <open3d/utility/FileSystem.h>

#include "geometry/VoxelBlockGrid.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tio = open3d::t::io;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry {

static std::pair<o3c::Tensor, o3c::Tensor> BufferRadiusNeighbors(
		std::shared_ptr<o3c::HashMap>& hashmap,
		const o3c::Tensor& active_buf_indices) {
	// Fixed radius search for spatially hashed voxel blocks.
	// A generalization will be implementing dense/sparse fixed radius
	// search with coordinates as hashmap keys.
	o3c::Tensor key_buffer_int3_tensor = hashmap->GetKeyTensor();

	o3c::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
			{active_buf_indices.To(o3c::Int64)});
	int64_t n = active_keys.GetShape()[0];

	// Fill in radius nearest neighbors.
	o3c::Tensor keys_nb({27, n, 3}, o3c::Int32, hashmap->GetDevice());
	for (int nb = 0; nb < 27; ++nb) {
		int dz = nb / 9;
		int dy = (nb % 9) / 3;
		int dx = nb % 3;
		o3c::Tensor dt =
				o3c::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1}, {1, 3},
				            o3c::Int32, hashmap->GetDevice());
		keys_nb[nb] = active_keys + dt;
	}
	keys_nb = keys_nb.View({27 * n, 3});

	o3c::Tensor buf_indices_nb, masks_nb;
	hashmap->Find(keys_nb, buf_indices_nb, masks_nb);
	return std::make_pair(buf_indices_nb.View({27, n, 1}),
	                      masks_nb.View({27, n, 1}));
}

o3tg::TensorMap VoxelBlockGrid::ConstructTensorMap(
		const o3c::HashMap& block_hashmap,
		std::unordered_map<std::string, int> name_attr_map) {
	o3tg::TensorMap tensor_map("tsdf");
	for (auto& v: name_attr_map) {
		std::string name = v.first;
		int buf_idx = v.second;
		tensor_map[name] = block_hashmap.GetValueTensor(buf_idx);
	}
	return tensor_map;
}

VoxelBlockGrid::VoxelBlockGrid(
		const std::vector<std::string>& attr_names,
		const std::vector<o3c::Dtype>& attr_dtypes,
		const std::vector<o3c::SizeVector>& attr_channels,
		float voxel_size,
		int64_t block_resolution,
		int64_t block_count,
		const o3c::Device& device,
		const o3c::HashBackendType& backend)
		: voxel_size_(voxel_size), block_resolution_(block_resolution) {

	// Sanity check
	if (voxel_size <= 0) {
		utility::LogError("voxel size must be positive, but got {}",
		                  voxel_size);
	}
	if (block_resolution <= 0) {
		utility::LogError("block resolution must be positive, but got {}",
		                  block_resolution);
	}


	// Check property lengths
	size_t n_attrs = attr_names.size();
	if (attr_dtypes.size() != n_attrs) {
		utility::LogError(
				"Number of attribute dtypes ({}) mismatch with names ({}).",
				attr_dtypes.size(), n_attrs);
	}
	if (attr_channels.size() != n_attrs) {
		utility::LogError(
				"Number of attribute channels ({}) mismatch with names ({}).",
				attr_channels.size(), n_attrs);
	}


	// Specify block element shapes and attribute names.
	std::vector<o3c::SizeVector> attr_element_shapes;
	o3c::SizeVector block_shape{block_resolution, block_resolution,
	                            block_resolution};
	for (size_t i = 0; i < n_attrs; ++i) {
		// Construct element shapes.
		o3c::SizeVector attr_channel = attr_channels[i];
		o3c::SizeVector block_shape_copy = block_shape;
		block_shape_copy.insert(block_shape_copy.end(), attr_channel.begin(),
		                        attr_channel.end());
		attr_element_shapes.emplace_back(block_shape_copy);

		// Used for easier accessing via attribute names.
		name_attr_map_[attr_names[i]] = i;
	}


	block_hashmap_ = std::make_shared<o3c::HashMap>(
			block_count, o3c::Int32, o3c::SizeVector{3}, attr_dtypes,
			attr_element_shapes, device, backend);
}

o3c::Tensor VoxelBlockGrid::GetAttribute(const std::string& attr_name) const {
	AssertInitialized();
	if (name_attr_map_.count(attr_name) == 0) {
		utility::LogWarning("Attribute {} not found, return empty tensor.",
		                    attr_name);
		return o3c::Tensor();
	}
	int buffer_idx = name_attr_map_.at(attr_name);
	return block_hashmap_->GetValueTensor(buffer_idx);
}

o3c::Tensor VoxelBlockGrid::GetVoxelCoordinates(
		const o3c::Tensor& voxel_indices) const {
	AssertInitialized();
	o3c::Tensor key_tensor = block_hashmap_->GetKeyTensor();

	o3c::Tensor voxel_coords =
			key_tensor.IndexGet({voxel_indices[0]}).T().To(o3c::Int64) *
			block_resolution_;
	voxel_coords[0] += voxel_indices[1];
	voxel_coords[1] += voxel_indices[2];
	voxel_coords[2] += voxel_indices[3];

	return voxel_coords;
}

o3c::Tensor VoxelBlockGrid::GetVoxelIndices(
		const o3c::Tensor& buf_indices) const {
	AssertInitialized();
	o3c::Device device = block_hashmap_->GetDevice();

	int64_t n_blocks = buf_indices.GetLength();

	int64_t resolution = block_resolution_;
	int64_t resolution2 = resolution * resolution;
	int64_t resolution3 = resolution2 * resolution;

	// Non-kernel version.
	/// TODO: Check if kernel version is necessary.
	o3c::Tensor linear_coordinates = o3c::Tensor::Arange(
			0, n_blocks * resolution3, 1, o3c::Int64, device);

	o3c::Tensor block_idx = linear_coordinates / resolution3;
	o3c::Tensor remainder = linear_coordinates - block_idx * resolution3;

	/// operator % is not supported now
	o3c::Tensor voxel_z = remainder / resolution2;
	remainder = remainder - voxel_z * resolution2;
	o3c::Tensor voxel_y = remainder / resolution;
	o3c::Tensor voxel_x = remainder - voxel_y * resolution;

	o3c::Tensor voxel_indices = o3c::Tensor({4, n_blocks * resolution3},
	                                        o3c::Dtype::Int64, device);
	voxel_indices[0] = buf_indices.IndexGet({block_idx}).To(o3c::Dtype::Int64);
	voxel_indices[1] = voxel_x;
	voxel_indices[2] = voxel_y;
	voxel_indices[3] = voxel_z;

	return voxel_indices;
}

o3c::Tensor VoxelBlockGrid::GetVoxelIndices() const {
	AssertInitialized();
	return GetVoxelIndices(block_hashmap_->GetActiveIndices());
}

std::pair<o3c::Tensor, o3c::Tensor>
VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices() {
	AssertInitialized();
	return GetVoxelCoordinatesAndFlattenedIndices(
			block_hashmap_->GetActiveIndices());
}

std::pair<o3c::Tensor, o3c::Tensor>
VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices(
		const o3c::Tensor& buf_indices) {
	AssertInitialized();
	// (N x resolution^3, 3) Float32; (N x resolution^3, 1) Int64
	int64_t n = buf_indices.GetLength();

	int64_t resolution3 =
			block_resolution_ * block_resolution_ * block_resolution_;

	o3c::Device device = block_hashmap_->GetDevice();
	o3c::Tensor voxel_coords({n * resolution3, 3}, o3c::Float32, device);
	o3c::Tensor flattened_indices({n * resolution3}, o3c::Int64, device);

	o3tg::kernel::voxel_grid::GetVoxelCoordinatesAndFlattenedIndices(
			buf_indices, block_hashmap_->GetKeyTensor(), voxel_coords,
			flattened_indices, block_resolution_, voxel_size_);
	return std::make_pair(voxel_coords, flattened_indices);
}

o3c::Tensor VoxelBlockGrid::GetUniqueBlockCoordinates(
		const o3tg::Image& depth,
		const o3c::Tensor& intrinsic,
		const o3c::Tensor& extrinsic,
		float depth_scale,
		float depth_max,
		float trunc_voxel_multiplier) {
	AssertInitialized();
	o3tg::CheckDepthTensor(depth.AsTensor());
	o3tg::CheckIntrinsicTensor(intrinsic);
	o3tg::CheckExtrinsicTensor(extrinsic);

	const int64_t down_factor = 4;
	const int64_t est_sample_multiplier = 4;
	if (frustum_hashmap_ == nullptr) {
		int64_t capacity = (depth.GetCols() / down_factor) *
		                   (depth.GetRows() / down_factor) *
		                   est_sample_multiplier;
		frustum_hashmap_ = std::make_shared<o3c::HashMap>(
				capacity, o3c::Int32, o3c::SizeVector{3}, o3c::Int32,
				o3c::SizeVector{1}, block_hashmap_->GetDevice());
	} else {
		frustum_hashmap_->Clear();
	}

	o3c::Tensor block_coords;
	o3tg::kernel::voxel_grid::DepthTouch(frustum_hashmap_, depth.AsTensor(),
	                                     intrinsic, extrinsic, block_coords,
	                                     block_resolution_, voxel_size_,
	                                     voxel_size_ * trunc_voxel_multiplier,
	                                     depth_scale, depth_max, down_factor);

	return block_coords;
}

o3c::Tensor VoxelBlockGrid::GetUniqueBlockCoordinates(
		const o3tg::PointCloud& pcd, float trunc_voxel_multiplier) {
	AssertInitialized();
	o3c::Tensor positions = pcd.GetPointPositions();

	const int64_t est_neighbor_multiplier = 8;
	if (frustum_hashmap_ == nullptr) {
		int64_t capacity = positions.GetLength() * est_neighbor_multiplier;
		frustum_hashmap_ = std::make_shared<o3c::HashMap>(
				capacity, o3c::Int32, o3c::SizeVector{3}, o3c::Int32,
				o3c::SizeVector{1}, block_hashmap_->GetDevice());
	} else {
		frustum_hashmap_->Clear();
	}

	o3c::Tensor block_coords;
	o3tg::kernel::voxel_grid::PointCloudTouch(
			frustum_hashmap_, positions, block_coords, block_resolution_,
			voxel_size_, voxel_size_ * trunc_voxel_multiplier);
	return block_coords;
}

void VoxelBlockGrid::Integrate(const o3c::Tensor& block_coords,
                               const o3tg::Image& depth,
                               const o3c::Tensor& intrinsic,
                               const o3c::Tensor& extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
	Integrate(block_coords, depth, o3tg::Image(), intrinsic, intrinsic, extrinsic,
	          depth_scale, depth_max, trunc_voxel_multiplier);
}

void VoxelBlockGrid::Integrate(const o3c::Tensor& block_coords,
                               const o3tg::Image& depth,
                               const o3tg::Image& color,
                               const o3c::Tensor& intrinsic,
                               const o3c::Tensor& extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
	Integrate(block_coords, depth, color, intrinsic, intrinsic, extrinsic,
	          depth_scale, depth_max, trunc_voxel_multiplier);
}

void VoxelBlockGrid::Integrate(const o3c::Tensor& block_coords,
                               const o3tg::Image& depth,
                               const o3tg::Image& color,
                               const o3c::Tensor& depth_intrinsic,
                               const o3c::Tensor& color_intrinsic,
                               const o3c::Tensor& extrinsic,
                               float depth_scale,
                               float depth_max,
                               float trunc_voxel_multiplier) {
	AssertInitialized();
	bool integrate_color = color.AsTensor().NumElements() > 0;

	o3tg::CheckBlockCoorinates(block_coords);
	o3tg::CheckDepthTensor(depth.AsTensor());
	if (integrate_color) {
		o3tg::CheckColorTensor(color.AsTensor());
	}
	o3tg::CheckIntrinsicTensor(depth_intrinsic);
	o3tg::CheckIntrinsicTensor(color_intrinsic);
	o3tg::CheckExtrinsicTensor(extrinsic);

	o3c::Tensor buf_indices, masks;
	block_hashmap_->Activate(block_coords, buf_indices, masks);
	block_hashmap_->Find(block_coords, buf_indices, masks);

	o3c::Tensor block_keys = block_hashmap_->GetKeyTensor();
	o3tg::TensorMap block_value_map =
			ConstructTensorMap(*block_hashmap_, name_attr_map_);

	o3tg::kernel::voxel_grid::Integrate(
			depth.AsTensor(), color.AsTensor(), buf_indices, block_keys,
			block_value_map, depth_intrinsic, color_intrinsic, extrinsic,
			block_resolution_, voxel_size_,
			voxel_size_ * trunc_voxel_multiplier, depth_scale, depth_max);
}

o3tg::TensorMap VoxelBlockGrid::RayCast(const o3c::Tensor& block_coords,
                                        const o3c::Tensor& intrinsic,
                                        const o3c::Tensor& extrinsic,
                                        int width,
                                        int height,
                                        const std::vector<std::string> attrs,
                                        float depth_scale,
                                        float depth_min,
                                        float depth_max,
                                        float weight_threshold,
                                        float trunc_voxel_multiplier,
                                        int range_map_down_factor) {
	AssertInitialized();
	o3tg::CheckBlockCoorinates(block_coords);
	o3tg::CheckIntrinsicTensor(intrinsic);
	o3tg::CheckExtrinsicTensor(extrinsic);

	// Extrinsic: world to camera -> pose: camera to world
	o3c::Device device = block_hashmap_->GetDevice();

	o3c::Tensor range_minmax_map;
	o3tg::kernel::voxel_grid::EstimateRange(block_coords, range_minmax_map, intrinsic,
	                                        extrinsic, height, width,
	                                        range_map_down_factor, block_resolution_,
	                                        voxel_size_, depth_min, depth_max);

	static const std::unordered_map<std::string, int> kAttrChannelMap = {
			// Conventional rendering
			{"vertex",          3},
			{"normal",          3},
			{"depth",           1},
			{"color",           3},
			// Diff rendering
			// Each pixel corresponds to info at 8 neighbor grid points
			{"index",           8},
			{"mask",            8},
			{"interp_ratio",    8},
			{"interp_ratio_dx", 8},
			{"interp_ratio_dy", 8},
			{"interp_ratio_dz", 8}};

	auto get_dtype = [&](const std::string& attr_name) -> o3c::Dtype {
		if (attr_name == "mask") {
			return o3c::Dtype::Bool;
		} else if (attr_name == "index") {
			return o3c::Dtype::Int64;
		} else {
			return o3c::Dtype::Float32;
		}
	};

	o3tg::TensorMap renderings_map("range");
	renderings_map["range"] = range_minmax_map;
	for (const auto& attr: attrs) {
		if (kAttrChannelMap.count(attr) == 0) {
			utility::LogError(
					"Unsupported attribute {}, please implement customized ray "
					"casting.");
		}
		int channel = kAttrChannelMap.at(attr);
		o3c::Dtype dtype = get_dtype(attr);
		renderings_map[attr] =
				o3c::Tensor({height, width, channel}, dtype, device);
	}

	o3tg::TensorMap block_value_map =
			ConstructTensorMap(*block_hashmap_, name_attr_map_);
	o3tg::kernel::voxel_grid::RayCast(
			block_hashmap_, block_value_map, range_minmax_map, renderings_map,
			intrinsic, extrinsic, height, width, block_resolution_, voxel_size_,
			depth_scale, depth_min, depth_max, weight_threshold,
			trunc_voxel_multiplier, range_map_down_factor);

	return renderings_map;
}

o3tg::PointCloud VoxelBlockGrid::ExtractPointCloud(float weight_threshold,
                                                   int estimated_point_number) {
	AssertInitialized();
	o3c::Tensor active_buf_indices;
	block_hashmap_->GetActiveIndices(active_buf_indices);

	o3c::Tensor active_nb_buf_indices, active_nb_masks;
	std::tie(active_nb_buf_indices, active_nb_masks) =
			BufferRadiusNeighbors(block_hashmap_, active_buf_indices);

	// Extract points around zero-crossings.
	o3c::Tensor points, normals, colors;

	o3c::Tensor block_keys = block_hashmap_->GetKeyTensor();
	o3tg::TensorMap block_value_map =
			ConstructTensorMap(*block_hashmap_, name_attr_map_);
	o3tg::kernel::voxel_grid::ExtractPointCloud(
			active_buf_indices, active_nb_buf_indices, active_nb_masks,
			block_keys, block_value_map, points, normals, colors,
			block_resolution_, voxel_size_, weight_threshold,
			estimated_point_number);

	auto pcd = o3tg::PointCloud(points.Slice(0, 0, estimated_point_number));
	pcd.SetPointNormals(normals.Slice(0, 0, estimated_point_number));

	if (colors.GetLength() == normals.GetLength()) {
		pcd.SetPointColors(colors.Slice(0, 0, estimated_point_number));
	}

	return pcd;
}

o3tg::TriangleMesh VoxelBlockGrid::ExtractTriangleMesh(float weight_threshold,
                                                       int estimated_vertex_number) {
	AssertInitialized();
	o3c::Tensor active_buf_indices_i32 = block_hashmap_->GetActiveIndices();
	o3c::Tensor active_nb_buf_indices, active_nb_masks;
	std::tie(active_nb_buf_indices, active_nb_masks) =
			BufferRadiusNeighbors(block_hashmap_, active_buf_indices_i32);

	o3c::Device device = block_hashmap_->GetDevice();
	// Map active indices to [0, num_blocks] to be allocated for surface mesh.
	int64_t num_blocks = block_hashmap_->Size();
	o3c::Tensor inverse_index_map({block_hashmap_->GetCapacity()}, o3c::Int32,
	                              device);
	o3c::Tensor iota_map =
			o3c::Tensor::Arange(0, num_blocks, 1, o3c::Int32, device);
	inverse_index_map.IndexSet({active_buf_indices_i32.To(o3c::Int64)},
	                           iota_map);

	o3c::Tensor vertices, triangles, vertex_normals, vertex_colors;

	o3c::Tensor block_keys = block_hashmap_->GetKeyTensor();
	o3tg::TensorMap block_value_map =
			ConstructTensorMap(*block_hashmap_, name_attr_map_);
	o3tg::kernel::voxel_grid::ExtractTriangleMesh(
			active_buf_indices_i32, inverse_index_map, active_nb_buf_indices,
			active_nb_masks, block_keys, block_value_map, vertices, triangles,
			vertex_normals, vertex_colors, block_resolution_, voxel_size_,
			weight_threshold, estimated_vertex_number);

	o3tg::TriangleMesh mesh(vertices, triangles);
	mesh.SetVertexNormals(vertex_normals);
	if (vertex_colors.GetLength() == vertices.GetLength()) {
		mesh.SetVertexColors(vertex_colors);
	}

	return mesh;
}

void VoxelBlockGrid::Save(const std::string& file_name) const {
	AssertInitialized();
	// TODO(wei): provide 'GetActiveKeyValues' functionality.
	o3c::Tensor keys = block_hashmap_->GetKeyTensor();
	std::vector<o3c::Tensor> values = block_hashmap_->GetValueTensors();

	o3c::Device host("CPU:0");

	o3c::Tensor active_buf_indices_i32 = block_hashmap_->GetActiveIndices();
	o3c::Tensor active_indices = active_buf_indices_i32.To(o3c::Int64);

	std::unordered_map<std::string, o3c::Tensor> output;

	// Save name attributes
	output.emplace("voxel_size", o3c::Tensor(std::vector<float>{voxel_size_},
	                                         {1}, o3c::Float32, host));
	output.emplace("block_resolution",
	               o3c::Tensor(std::vector<int64_t>{block_resolution_}, {1},
	                           o3c::Int64, host));
	// Placeholder
	output.emplace(block_hashmap_->GetDevice().ToString(),
	               o3c::Tensor::Zeros({}, o3c::Dtype::UInt8, host));

	for (auto& it: name_attr_map_) {
		// Workaround, as we don't support char tensors now.
		output.emplace(fmt::format("attr_name_{}", it.first),
		               o3c::Tensor(std::vector<int>{it.second}, {1},
		                           o3c::Int32, host));
	}

	// Save keys
	o3c::Tensor active_keys = keys.IndexGet({active_indices}).To(host);
	output.emplace("key", active_keys);

	// Save SoA values and name attributes
	for (auto& it: name_attr_map_) {
		int value_id = it.second;
		o3c::Tensor active_value_i =
				values[value_id].IndexGet({active_indices}).To(host);
		output.emplace(fmt::format("value_{:03d}", value_id), active_value_i);
	}

	std::string ext =
			utility::filesystem::GetFileExtensionInLowerCase(file_name);
	if (ext != "npz") {
		utility::LogWarning(
				"File name for a voxel grid should be with the extension "
				".npz. Saving to {}.npz",
				file_name);
		o3tio::WriteNpz(file_name + ".npz", output);
	} else {
		o3tio::WriteNpz(file_name, output);
	}
}

VoxelBlockGrid VoxelBlockGrid::Load(const std::string& file_name) {
	std::unordered_map<std::string, o3c::Tensor> tensor_map =
			o3tio::ReadNpz(file_name);

	std::string prefix = "attr_name_";
	std::unordered_map<int, std::string> inv_attr_map;

	std::string kCPU = "CPU";
	std::string kCUDA = "CUDA";

	std::string device_str = "CPU:0";
	for (auto& it: tensor_map) {
		if (!it.first.compare(0, prefix.size(), prefix)) {
			int value_id = it.second[0].Item<int>();
			inv_attr_map.emplace(value_id, it.first.substr(prefix.size()));
		}
		if (!it.first.compare(0, kCPU.size(), kCPU) ||
		    !it.first.compare(0, kCUDA.size(), kCUDA)) {
			device_str = it.first;
		}
	}
	if (inv_attr_map.size() == 0) {
		utility::LogError(
				"Attribute names not found, not a valid file for voxel block "
				"grids.");
	}

	o3c::Device device(device_str);

	std::vector<std::string> attr_names(inv_attr_map.size());

	std::vector<o3c::Tensor> soa_value_tensor(inv_attr_map.size());
	std::vector<o3c::Dtype> attr_dtypes(inv_attr_map.size());
	std::vector<o3c::SizeVector> attr_channels(inv_attr_map.size());

	// Not an ideal way to use an unordered map. Assume all the indices are
	// stored.
	for (auto& v: inv_attr_map) {
		int value_id = v.first;
		attr_names[value_id] = v.second;

		o3c::Tensor value_i =
				tensor_map.at(fmt::format("value_{:03d}", value_id));

		soa_value_tensor[value_id] = value_i.To(device);
		attr_dtypes[value_id] = value_i.GetDtype();

		o3c::SizeVector value_i_shape = value_i.GetShape();
		// capacity, res, res, res
		value_i_shape.erase(value_i_shape.begin(), value_i_shape.begin() + 4);
		attr_channels[value_id] = value_i_shape;
	}

	o3c::Tensor keys = tensor_map.at("key").To(device);
	float voxel_size = tensor_map.at("voxel_size")[0].Item<float>();
	int block_resolution = tensor_map.at("block_resolution")[0].Item<int64_t>();

	VoxelBlockGrid vbg(attr_names, attr_dtypes, attr_channels, voxel_size,
	                   block_resolution, keys.GetLength(), device);
	auto block_hashmap = vbg.GetHashMap();
	block_hashmap.Insert(keys, soa_value_tensor);
	return vbg;
}

void VoxelBlockGrid::AssertInitialized() const {
	if (block_hashmap_ == nullptr) {
		utility::LogError("VoxelBlockGrid not initialized.");
	}
}

o3c::Device VoxelBlockGrid::GetDevice() const {
	return this->block_hashmap_->GetDevice();
}

float VoxelBlockGrid::GetVoxelSize() const {
	return this->voxel_size_;
}

int64_t VoxelBlockGrid::GetBlockResolution() const {
	return this->block_resolution_;
}

int64_t VoxelBlockGrid::GetBlockCount() const {
	return this->block_hashmap_->GetCapacity();
}


VoxelBlockGrid VoxelBlockGrid::To(const open3d::core::Device& device, bool force_copy /*= false*/) const {
	if (!force_copy && this->GetDevice() == device) {
		return *this;
	}
	VoxelBlockGrid grid_copy;
	grid_copy.voxel_size_ = this->voxel_size_;
	grid_copy.block_resolution_ = this->block_resolution_;
	if (this->block_hashmap_ != nullptr) {
		grid_copy.block_hashmap_ = std::make_shared<o3c::HashMap>(this->block_hashmap_->To(device));
	} else {
		grid_copy.block_hashmap_ = nullptr;
	}
	if (this->frustum_hashmap_ != nullptr){
		grid_copy.frustum_hashmap_ = std::make_shared<o3c::HashMap>(this->frustum_hashmap_->To(device));
	}else {
		grid_copy.frustum_hashmap_ = nullptr;
	}
	grid_copy.name_attr_map_ = this->name_attr_map_;
	return grid_copy;
}


bool operator==(const VoxelBlockGrid& lhs, const VoxelBlockGrid& rhs) {
	if (lhs.GetDevice() != rhs.GetDevice()) return false;
	// voxel size, block resolution
	if (lhs.voxel_size_ != rhs.voxel_size_) return false;
	if (lhs.block_resolution_ != rhs.block_resolution_) return false;
	// name attribute count and map
	if (lhs.name_attr_map_.size() != rhs.name_attr_map_.size()) return false;
	for (auto&& [lhs_attribute_name, lhs_attribute_index]: lhs.name_attr_map_) {
		if (rhs.name_attr_map_.find(lhs_attribute_name) == rhs.name_attr_map_.end()) return false;
		if (rhs.name_attr_map_.at(lhs_attribute_name) != lhs_attribute_index) return false;
	}
	if (lhs.block_hashmap_->GetCapacity() != rhs.block_hashmap_->GetCapacity()) return false;

	o3c::Tensor lhs_key_tensor = lhs.block_hashmap_->GetKeyTensor();
	o3c::Tensor lhs_active_indices = lhs.block_hashmap_->GetActiveIndices().To(o3c::Int64);
	o3c::Tensor lhs_active_keys = lhs_key_tensor.GetItem(o3c::TensorKey::IndexTensor(lhs_active_indices));

	o3c::Tensor rhs_matching_key_indices, rhs_key_matches_found_mask;
	rhs.block_hashmap_->Find(lhs_active_keys, rhs_matching_key_indices, rhs_key_matches_found_mask);
	int64_t total_lhs_keys_found_in_rhs = rhs_key_matches_found_mask.To(o3c::Int64).Sum({0}).ToFlatVector<int64_t>()[0];

	if (total_lhs_keys_found_in_rhs != lhs_active_keys.GetLength()) return false;

	for (auto&& [lhs_attribute_name, lhs_attribute_index]: lhs.name_attr_map_) { // NOLINT(readability-use-anyofallof)
		if (!lhs.block_hashmap_->GetValueTensor(lhs_attribute_index).GetItem(o3c::TensorKey::IndexTensor(lhs_active_indices))
				.AllEqual(
						rhs.block_hashmap_->GetValueTensor(rhs.name_attr_map_.at(lhs_attribute_name))
								.GetItem(o3c::TensorKey::IndexTensor(rhs_matching_key_indices.To(o3c::Int64)))
				)
				) {
			return false;
		}
	}
	return true;
}


}  // namespace nnrt::geometry

