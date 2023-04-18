//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 4/17/23.
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
#include <open3d/core/ParallelFor.h>
#include <Eigen/Dense>

// local includes
#include "alignment/functional/kernel/AssociateFacesWithAnchors.h"
#include "core/platform_independence/Qualifiers.h"
#include "alignment/functional/FaceNodeAnchors.h"
#include "geometry/functional/kernel/Defines.h"

namespace o3c = open3d::core;


namespace nnrt::alignment::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType>
void AssociateFacesWithAnchors(
		std::shared_ptr<open3d::core::Blob>& face_node_anchors,
		open3d::core::Tensor& face_anchor_counts,
		const open3d::core::Tensor& face_vertex_indices,
		const open3d::core::Tensor& warp_vertex_anchors
) {

	// === checks, counts ====
	int64_t face_count = face_vertex_indices.GetShape(0);
	o3c::AssertTensorShape(face_vertex_indices, { face_count, 3 });
	o3c::Device device = face_vertex_indices.GetDevice();
	o3c::AssertTensorDevice(warp_vertex_anchors, device);
	int64_t vertex_count = warp_vertex_anchors.GetShape(0);
	int64_t anchor_per_vertex_count = warp_vertex_anchors.GetShape(1);
	o3c::AssertTensorShape(warp_vertex_anchors, { vertex_count, anchor_per_vertex_count });

	o3c::AssertTensorDtype(face_vertex_indices, o3c::Int64);
	o3c::AssertTensorDtype(warp_vertex_anchors, o3c::Int32);

	const int max_face_anchor_count = 3 * static_cast<int>(anchor_per_vertex_count);

	// === input pointers ===
	auto face_vertex_index_data = face_vertex_indices.GetDataPtr<int64_t>();
	auto warp_vertex_anchor_data = warp_vertex_anchors.GetDataPtr<int32_t>();


	// === output structures and pointers ===
	face_anchor_counts = o3c::Tensor::Zeros({face_count}, o3c::Int32, device);
	const int64_t face_node_anchor_stride = 3 * MAX_ANCHOR_COUNT;
	face_node_anchors = std::make_shared<open3d::core::Blob>(face_count * face_node_anchor_stride * sizeof(FaceNodeAnchors), device);
	auto face_node_anchor_data = reinterpret_cast<FaceNodeAnchors*>(face_node_anchors->GetDataPtr());
	auto face_anchor_count_data = face_anchor_counts.GetDataPtr<int32_t>();

	o3c::ParallelFor(
			device, face_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t i_face) {
				Eigen::Map<const Eigen::RowVector3<int64_t>> vertex_indices(face_vertex_index_data + i_face * 3);
				int current_face_anchor_count = 0;
				FaceNodeAnchors current_face_anchor_data[face_node_anchor_stride];

				for (int i_face_vertex = 0; i_face_vertex < 3; i_face_vertex++) {
					int64_t i_vertex = vertex_indices(i_face_vertex);
					for (int i_vertex_anchor = 0; i_vertex_anchor < anchor_per_vertex_count; i_vertex_anchor++) {
						auto i_node = warp_vertex_anchor_data[i_vertex * anchor_per_vertex_count + i_vertex_anchor];
						if (i_node == -1) {
							// -1 is the sentinel value for anchors
							continue;
						}
						int i_face_anchor = 0;
						int inspected_anchor_node_index = current_face_anchor_data[i_face_anchor].node_index;
						// find (if any) matching node among anchors of previous vertices
						while (
								inspected_anchor_node_index != i_node && // found match in previous matches
								inspected_anchor_node_index != -1 && // end of filled list reached
								i_face_anchor + 1 < max_face_anchor_count// end of list reached)
								) {
							i_face_anchor++;
							inspected_anchor_node_index = current_face_anchor_data[i_face_anchor].node_index;
						}

						auto& face_anchor = current_face_anchor_data[i_face_anchor];
						if (inspected_anchor_node_index != i_node) { // unique node found
							current_face_anchor_count++; // tally per pixel
							face_anchor.node_index = i_node;
						}
						face_anchor.vertices[i_face_vertex] = i_vertex;
						face_anchor.vertex_anchor_indices[i_face_vertex] = i_vertex_anchor;
					}
				}
				face_anchor_count_data[i_face] = current_face_anchor_count;
				memcpy(face_node_anchor_data + i_face * face_node_anchor_stride, current_face_anchor_data, face_node_anchor_stride * sizeof(FaceNodeAnchors));
			}
	);
}

} // namespace nnrt::alignment::functional::kernel