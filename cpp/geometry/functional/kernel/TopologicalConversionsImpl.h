//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 5/1/23.
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
#pragma once
// stdlib includes

// third-party includes
#include <open3d/utility/Logging.h>
#include <open3d/core/Dispatch.h>
#include <open3d/core/ParallelFor.h>
#include "core/functional/kernel/BubbleSort.h"

// local includes
#include "geometry/functional/kernel/TopologicalConversions.h"
#include "core/platform_independence/AtomicCounterArray.h"
#include "core/platform_independence/Qualifiers.h"

namespace utility = open3d::utility;
namespace o3c = open3d::core;
namespace nnrt::geometry::functional::kernel {

template<typename TVertexIndexType, open3d::core::Device::DeviceType TDeviceType>
void
MeshToAdjacencyArray_Generic(
		open3d::core::Tensor& adjacency_array,
		const open3d::core::Tensor& triangle_indices,
		int max_expected_vertex_degree,
		int64_t vertex_count
) {
	const TVertexIndexType* triangle_index_data = triangle_indices.GetDataPtr<TVertexIndexType>();
	o3c::Device device = triangle_indices.GetDevice();
	int64_t triangle_count = triangle_indices.GetLength();
	int adjacencies_with_duplicates_step = max_expected_vertex_degree * 2;
	o3c::Tensor vertex_adjacencies_with_duplicates({vertex_count, adjacencies_with_duplicates_step}, triangle_indices.GetDtype(), device);
	TVertexIndexType* adjacency_data_with_duplicates = vertex_adjacencies_with_duplicates.GetDataPtr<TVertexIndexType>();
	core::AtomicCounterArray<TDeviceType> vertex_adjacency_counts(vertex_count);

	// compile adjacency lists. Note: these might (and, almost certainly, will) have duplicates, since two triangles often share the same edge.
	o3c::ParallelFor(
			device, triangle_count * 3,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t triangle_index = workload_idx / 3;
				int64_t index0_in_triangle = workload_idx % 3;
				TVertexIndexType index0 = triangle_index_data[triangle_index * 3 + index0_in_triangle];
				TVertexIndexType index1 = triangle_index_data[triangle_index * 3 + ((index0_in_triangle + 1) % 3)];
				TVertexIndexType source_index, target_index;
				if (index0 < index1) {
					source_index = index0;
					target_index = index1;
				} else {
					source_index = index1;
					target_index = index0;
				}
				int i_adjacency = vertex_adjacency_counts.FetchAdd(static_cast<int>(source_index), 1);
				if (i_adjacency > adjacencies_with_duplicates_step) {
					printf("Warning: vertex degree appears to be greater than the max_expected_vertex_degree argument can accommodate, "
					       "the result of adjacency array generation will be incomplete. Please re-run with this argument increased.");
				} else {
					adjacency_data_with_duplicates[static_cast<int>(source_index) * adjacencies_with_duplicates_step + i_adjacency] = target_index;
				}
			}
	);

	adjacency_array = o3c::Tensor({vertex_count, max_expected_vertex_degree}, triangle_indices.GetDtype(), device);
	adjacency_array.Fill(-1);
	TVertexIndexType sentinel = std::numeric_limits<TVertexIndexType>::max();

	TVertexIndexType* adjacency_data = adjacency_array.GetDataPtr<TVertexIndexType>();
	// sort adjacency lists and weed out the duplicates.
	o3c::ParallelFor(
			device, vertex_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_vertex) {
				int vertex_count = vertex_adjacency_counts.GetCount(i_vertex);
				TVertexIndexType* vertex_raw_adjacencies = adjacency_data_with_duplicates + i_vertex * adjacencies_with_duplicates_step;
				//TODO: replace BubbleSort with thrust::sort?
#ifdef __CUDACC__
				core::functional::kernel::BubbleSort(vertex_raw_adjacencies, vertex_count);
#else
				std::sort(vertex_raw_adjacencies, vertex_raw_adjacencies + vertex_count);
#endif
				TVertexIndexType* vertex_unique_adjacencies = adjacency_data + i_vertex * max_expected_vertex_degree;
				TVertexIndexType previous_adjacency = sentinel;
				int unique_adjacency_count = 0;
				for (int i_raw_adjacency = 0; i_raw_adjacency < vertex_count; i_raw_adjacency++) {
					TVertexIndexType raw_adjacency = vertex_raw_adjacencies[i_raw_adjacency];
					if (raw_adjacency != previous_adjacency) {
						previous_adjacency = raw_adjacency;
						if (unique_adjacency_count > max_expected_vertex_degree) {
							printf("Warning: vertex degree appears to be greater than the max_expected_vertex_degree argument can accommodate, "
							       "the result of adjacency array generation will be incomplete. Please re-run with this argument increased.");
						} else {
							vertex_unique_adjacencies[unique_adjacency_count] = raw_adjacency;
							unique_adjacency_count++;
						}
					}
				}
			}
	);

}


template<open3d::core::Device::DeviceType TDeviceType>
void MeshToAdjacencyArray(open3d::core::Tensor& adjacency_array, const open3d::t::geometry::TriangleMesh& mesh, int max_expected_vertex_degree) {
	if (!mesh.HasTriangleIndices()) {
		utility::LogError("Procedure can only convert a mesh which has triangle indices defined into an adjacency array.");
	}
	if (!mesh.HasVertexPositions()) {
		utility::LogError("Procedure can only convert a mesh which has vertex positions defined into an adjacency array.");
	}

	const o3c::Tensor& triangle_indices = mesh.GetTriangleIndices();
	o3c::AssertTensorDtypes(triangle_indices, { o3c::Int32, o3c::Int64 });
	int64_t vertex_count = mesh.GetVertexPositions().GetLength();


	DISPATCH_DTYPE_TO_TEMPLATE(
			triangle_indices.GetDtype(),
			[&] {
				MeshToAdjacencyArray_Generic<scalar_t, TDeviceType>(adjacency_array, triangle_indices, max_expected_vertex_degree, vertex_count);
			}
	);
}

} // namespace nnrt::geometry::functional::kernel