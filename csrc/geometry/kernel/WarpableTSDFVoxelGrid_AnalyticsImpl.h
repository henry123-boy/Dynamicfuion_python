//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/23/21.
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

#include <Eigen/Geometry>

#include <open3d/core/Tensor.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>

using namespace open3d;
namespace o3c = open3d::core;
using namespace open3d::t::geometry::kernel;
using namespace open3d::t::geometry::kernel::tsdf;


namespace nnrt {
namespace geometry {
namespace kernel {
namespace tsdf {


#if defined(__CUDACC__)
void ExtractVoxelCentersCUDA
#else
void ExtractVoxelCentersCPU
#endif
		(const o3c::Tensor& indices,
		 const o3c::Tensor& block_keys,
		 const o3c::Tensor& block_values,
		 o3c::Tensor& voxel_centers,
		 int64_t block_resolution, float voxel_size) {

	int64_t resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});

	int n_blocks = static_cast<int>(indices.GetLength());

	int64_t n_voxels = n_blocks * resolution3;
	// each voxel center will need three coordinates: n_voxels x 3
	voxel_centers = o3c::Tensor({n_voxels, 3}, o3c::Dtype::Float32,
	                             block_keys.GetDevice());

	// Real data indexers
	NDArrayIndexer voxel_centers_indexer(voxel_centers, 1);
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	// Plain array that does not require indexers
	const int64_t* indices_ptr = indices.GetDataPtr<int64_t>();

	// Go through voxels
	open3d::core::ParallelFor(
			indices.GetDevice(), n_voxels,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {

		// Natural index (0, N) ->
		//                        (workload_block_idx, voxel_index_in_block)
		int64_t workload_block_idx = workload_idx / resolution3;
		int64_t block_index = indices_ptr[workload_block_idx];
		int64_t voxel_index_in_block = workload_idx % resolution3;

		// block_index -> (x_block, y_block, z_block)
		int* block_key_ptr =
				block_keys_indexer.GetDataPtr<int>(block_index);
		int64_t x_block = static_cast<int64_t>(block_key_ptr[0]);
		int64_t y_block = static_cast<int64_t>(block_key_ptr[1]);
		int64_t z_block = static_cast<int64_t>(block_key_ptr[2]);

		// voxel_idx -> (x_voxel, y_voxel, z_voxel)
		int64_t x_voxel, y_voxel, z_voxel;
		voxel_indexer.WorkloadToCoord(voxel_index_in_block,
		                              &x_voxel, &y_voxel, &z_voxel);


		auto* voxel_center_pointer = voxel_centers_indexer.GetDataPtr<float>(workload_idx);

		voxel_center_pointer[0] = static_cast<float>(x_block * block_resolution + x_voxel) * voxel_size;
		voxel_center_pointer[1] = static_cast<float>(y_block * block_resolution + y_voxel) * voxel_size;
		voxel_center_pointer[2] = static_cast<float>(z_block * block_resolution + z_voxel) * voxel_size;
	}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

#if defined(__CUDACC__)
void ExtractTSDFValuesAndWeightsCUDA
#else
void ExtractTSDFValuesAndWeightsCPU
#endif
		(const o3c::Tensor& indices,
		 const o3c::Tensor& block_values,
		 o3c::Tensor& voxel_values,
		 int64_t block_resolution) {

	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});

	int n_blocks = static_cast<int>(indices.GetLength());


	int64_t n_voxels = n_blocks * block_resolution3;
	// each voxel output will need a TSDF value and a weight value: n_voxels x 2
	voxel_values = o3c::Tensor::Zeros({n_voxels, 2}, o3c::Dtype::Float32,
	                                   block_values.GetDevice());

	// Real data indexers
	NDArrayIndexer voxel_values_indexer(voxel_values, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

	// Plain arrays that does not require indexers
	const auto* indices_ptr = indices.GetDataPtr<int64_t>();


	//  Go through voxels
//@formatter:off
	DISPATCH_BYTESIZE_TO_VOXEL(
			voxel_block_buffer_indexer.ElementByteSize(),
			[&]() {
				open3d::core::ParallelFor(
						indices.GetDevice(), n_voxels,
						[=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
				// Natural index (0, N) ->
				//                        (workload_block_idx, voxel_index_in_block)
				int64_t block_idx = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;

				// voxel_idx -> (x_voxel, y_voxel, z_voxel)
				int64_t x_local, y_local, z_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block,
				                              &x_local, &y_local, &z_local);

				auto voxel_ptr = voxel_block_buffer_indexer
						.GetDataPtr<voxel_t>(x_local, y_local, z_local, block_idx);

				auto voxel_value_pointer = voxel_values_indexer.GetDataPtr<float>(workload_idx);

				voxel_value_pointer[0] = voxel_ptr->GetTSDF();
				voxel_value_pointer[1] = static_cast<float>(voxel_ptr->GetWeight());

			} // end lambda
				);
			}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}


#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ExtractValuesInExtentCUDA(
#else
void ExtractValuesInExtentCPU(
#endif
		int64_t min_x, int64_t min_y, int64_t min_z,
		int64_t max_x, int64_t max_y, int64_t max_z,
		const o3c::Tensor& block_indices,
		const o3c::Tensor& block_keys,
		const o3c::Tensor& block_values,
		o3c::Tensor& voxel_values,
		int64_t block_resolution) {

	int64_t block_resolution3 =
			block_resolution * block_resolution * block_resolution;

	// Shape / transform indexers, no data involved
	NDArrayIndexer voxel_indexer(
			{block_resolution, block_resolution, block_resolution});


	int n_blocks = static_cast<int>(block_indices.GetLength());

	int64_t output_range_x = max_x - min_x;
	int64_t output_range_y = max_y - min_y;
	int64_t output_range_z = max_z - min_z;

	int64_t n_voxels = n_blocks * block_resolution3;
	// each voxel center will need three coordinates: n_voxels x 3
	voxel_values = o3c::Tensor::Ones({output_range_x, output_range_y, output_range_z},
	                                  o3c::Dtype::Float32, block_keys.GetDevice());
	voxel_values *= -2.0f;

	// Real data indexers
	NDArrayIndexer voxel_value_indexer(voxel_values, 3);
	NDArrayIndexer block_keys_indexer(block_keys, 1);
	NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

	// Plain array that does not require indexers
	const auto* indices_ptr = block_indices.GetDataPtr<int64_t>();

//  Go through voxels
//@formatter:off
	DISPATCH_BYTESIZE_TO_VOXEL(
			voxel_block_buffer_indexer.ElementByteSize(),
			[&]() {
				open3d::core::ParallelFor(
						block_indices.GetDevice(), n_voxels,
						[=] OPEN3D_DEVICE(int64_t workload_idx) {
//@formatter:on
				// Natural index (0, N) ->
				//                    (workload_block_idx, voxel_index_in_block)
				int64_t block_index = indices_ptr[workload_idx / block_resolution3];
				int64_t voxel_index_in_block = workload_idx % block_resolution3;

				// block_index -> (x_block, y_block, z_block)
				int* block_key_ptr =
						block_keys_indexer.GetDataPtr<int>(block_index);
				auto x_block = static_cast<int64_t>(block_key_ptr[0]);
				auto y_block = static_cast<int64_t>(block_key_ptr[1]);
				auto z_block = static_cast<int64_t>(block_key_ptr[2]);

				// voxel_idx -> (x_voxel, y_voxel, z_voxel)
				int64_t x_voxel_local, y_voxel_local, z_voxel_local;
				voxel_indexer.WorkloadToCoord(voxel_index_in_block, &x_voxel_local, &y_voxel_local, &z_voxel_local);

				// at this point, (x_voxel, y_voxel, z_voxel) hold local
				// in-block coordinates. Compute the global voxel coordinates:
				int64_t x_voxel_global = x_block * block_resolution + x_voxel_local;
				int64_t y_voxel_global = y_block * block_resolution + y_voxel_local;
				int64_t z_voxel_global = z_block * block_resolution + z_voxel_local;

				int64_t x_voxel_out = x_voxel_global - min_x;
				int64_t y_voxel_out = y_voxel_global - min_y;
				int64_t z_voxel_out = z_voxel_global - min_z;

				if (x_voxel_out >= 0 && x_voxel_out < output_range_x &&
				    y_voxel_out >= 0 && y_voxel_out < output_range_y &&
				    z_voxel_out >= 0 && z_voxel_out < output_range_z) {
					auto* voxel_value_pointer =
							voxel_value_indexer.GetDataPtr<float>(
									x_voxel_out, y_voxel_out, z_voxel_out);

					auto voxel_pointer = voxel_block_buffer_indexer.GetDataPtr<voxel_t>(
							x_voxel_local, y_voxel_local, z_voxel_local, block_index);

					auto weight = voxel_pointer->GetWeight();

					if (weight > 0) {
						*voxel_value_pointer = voxel_pointer->GetTSDF();
					}
				}
			} // end element_kernel
				);
			}
	);
#if defined(__CUDACC__)
	OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

} // namespace tsdf
} // namespace kernel
} // namespace geometry
} // namespace nnrt
