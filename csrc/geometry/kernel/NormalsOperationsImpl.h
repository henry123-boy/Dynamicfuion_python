//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/21/22.
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
#pragma once

// stdlib
#include <atomic>

// 3rd party
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/core/TensorCheck.h>
#include <open3d/utility/Optional.h>
#include <Eigen/Dense>

// local
#include "geometry/kernel/NormalsOperations.h"
#include "core/PlatformIndependentAtomics.h"

namespace o3c = open3d::core;
namespace o3u = open3d::utility;
namespace o3gk = open3d::t::geometry::kernel;

namespace nnrt::geometry::kernel::mesh {

template<open3d::core::Device::DeviceType TDevice>
void ComputeTriangleNormals(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                            const open3d::core::Tensor& triangle_indices) {
	auto device = vertex_positions.GetDevice();
	o3c::AssertTensorDtype(vertex_positions, o3c::Float32);
	o3c::AssertTensorShape(vertex_positions, {o3u::nullopt, 3});
	o3c::AssertTensorDtype(triangle_indices, o3c::Int64);
	o3c::AssertTensorShape(triangle_indices, {o3u::nullopt, 3});


	triangle_normals = o3c::Tensor({triangle_indices.GetLength(), 3}, o3c::Float32, device);

	o3gk::NDArrayIndexer vertex_indexer(vertex_positions, 1);
	o3gk::NDArrayIndexer triangle_indexer(triangle_indices, 1);
	o3gk::NDArrayIndexer normal_indexer(triangle_normals, 1);

	o3c::ParallelFor(
			device, triangle_indices.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				Eigen::Map<Eigen::Matrix<int64_t, 1, 3>> triangle_vertex_indices(triangle_indexer.template GetDataPtr<int64_t>(workload_idx));

				Eigen::Map<Eigen::Vector3f> vertex0(vertex_indexer.template GetDataPtr<float>(triangle_vertex_indices(0)));
				Eigen::Map<Eigen::Vector3f> vertex1(vertex_indexer.template GetDataPtr<float>(triangle_vertex_indices(1)));
				Eigen::Map<Eigen::Vector3f> vertex2(vertex_indexer.template GetDataPtr<float>(triangle_vertex_indices(2)));

				Eigen::Vector3f v0_to_v1 = vertex1 - vertex0;
				Eigen::Vector3f v0_to_v2 = vertex2 - vertex0;

				Eigen::Map<Eigen::Vector3f> triangle_normal(normal_indexer.template GetDataPtr<float>(workload_idx));
				triangle_normal = v0_to_v1.template cross(v0_to_v2);
			}
	);
}

template<open3d::core::Device::DeviceType TDevice>
void NormalizeVectors3d(open3d::core::Tensor& vectors3f) {
	o3c::AssertTensorDtype(vectors3f, o3c::Float32);
	o3c::AssertTensorShape(vectors3f, {o3u::nullopt, 3});

	o3gk::NDArrayIndexer vector_indexer(vectors3f, 1);

	o3c::ParallelFor(
			vectors3f.GetDevice(), vectors3f.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				Eigen::Map<Eigen::Vector3f> vector3f(vector_indexer.GetDataPtr<float>(workload_idx));
				vector3f.normalize();
#ifndef __CUDACC__
				if (std::isnan(vector3f(0))) {
#else
					if (isnan(vector3f(0))){
#endif
					vector3f = Eigen::Vector3f(0.0, 0.0, 1.0);
				}
			}
	);
}

template<open3d::core::Device::DeviceType TDevice>
void ComputeVertexNormals(open3d::core::Tensor& vertex_normals, const open3d::core::Tensor& triangle_indices,
                          const open3d::core::Tensor& triangle_normals) {
	o3c::Device device = triangle_indices.GetDevice();

	o3c::AssertTensorDtype(triangle_indices, o3c::Int64);
	o3c::AssertTensorShape(triangle_indices, {o3u::nullopt, 3});
	o3c::AssertTensorDtype(triangle_normals, o3c::Float32);
	o3c::AssertTensorShape(triangle_normals, {o3u::nullopt, 3});


	o3gk::NDArrayIndexer vertex_normal_indexer(vertex_normals, 1);
	o3gk::NDArrayIndexer triangle_indexer(triangle_indices, 1);
	o3gk::NDArrayIndexer triangle_normal_indexer(triangle_normals, 1);

#ifndef __CUDACC__
	o3c::Blob atomic_vertex_normals_blob(static_cast<int64_t>(sizeof(std::atomic<float>)) * 3 * vertex_normals.GetLength(), device);
	auto* vertex_normals_base_ptr = reinterpret_cast<std::atomic<float>*>(atomic_vertex_normals_blob.GetDataPtr());
#else
	auto* vertex_normals_base_ptr = reinterpret_cast<float*>(vertex_normals.GetDataPtr());
#endif


#ifndef __CUDACC__
	o3c::ParallelFor(
			device, vertex_normals.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto atomic_vertex_normal_ptr = vertex_normals_base_ptr + workload_idx * 3;
				atomic_vertex_normal_ptr[0].store(0.f);
				atomic_vertex_normal_ptr[1].store(0.f);
				atomic_vertex_normal_ptr[2].store(0.f);
			}
	);
#endif

	o3c::ParallelFor(
			device, triangle_indices.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				Eigen::Map<Eigen::Matrix<int64_t, 1, 3>> triangle_vertex_indices(triangle_indexer.template GetDataPtr<int64_t>(workload_idx));
				Eigen::Map<Eigen::Vector3f> triangle_normal(triangle_normal_indexer.template GetDataPtr<float>(workload_idx));

				auto vertex0_normal_ptr = vertex_normals_base_ptr + triangle_vertex_indices(0) * 3;
				auto vertex1_normal_ptr = vertex_normals_base_ptr + triangle_vertex_indices(1) * 3;;
				auto vertex2_normal_ptr = vertex_normals_base_ptr + triangle_vertex_indices(2) * 3;;

				NNRT_ATOMIC_ADD(vertex0_normal_ptr, triangle_normal.x());
				NNRT_ATOMIC_ADD(vertex0_normal_ptr+1, triangle_normal.y());
				NNRT_ATOMIC_ADD(vertex0_normal_ptr+2, triangle_normal.z());

				NNRT_ATOMIC_ADD(vertex1_normal_ptr, triangle_normal.x());
				NNRT_ATOMIC_ADD(vertex1_normal_ptr+1, triangle_normal.y());
				NNRT_ATOMIC_ADD(vertex1_normal_ptr+2, triangle_normal.z());

				NNRT_ATOMIC_ADD(vertex2_normal_ptr, triangle_normal.x());
				NNRT_ATOMIC_ADD(vertex2_normal_ptr+1, triangle_normal.y());
				NNRT_ATOMIC_ADD(vertex2_normal_ptr+2, triangle_normal.z());
			}
	);

#ifndef __CUDACC__
	o3c::ParallelFor(
			device, vertex_normals.GetLength(),
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto atomic_vertex_normal_ptr = vertex_normals_base_ptr + workload_idx * 3;
				auto vertex_normal_ptr = vertex_normal_indexer.template GetDataPtr<float>(workload_idx);
				vertex_normal_ptr[0] = atomic_vertex_normal_ptr[0].load();
				vertex_normal_ptr[1] = atomic_vertex_normal_ptr[1].load();
				vertex_normal_ptr[2] = atomic_vertex_normal_ptr[2].load();
			}
	);
#endif
}

} // nnrt::geometry::kernel::mesh