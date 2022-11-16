//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 6/9/21.
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
#include "geometry/kernel/Warp3dPointsAndNormals.h"
#include "core/DeviceSelection.h"
#include <open3d/t/geometry/Utility.h>

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;

namespace nnrt::geometry::kernel::warp {

// region =========================== POINTS ONLY ====================================================================================================
// computes anchors on-the-fly, does not use node distance thresholding
void Warp3dPoints(
		open3d::core::Tensor& warped_vertices, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPoints<o3c::Device::DeviceType::CPU>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage, extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPoints<o3c::Device::DeviceType::CUDA>(
								warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage, extrinsics
						);
				);
			}
	);
}

// computes anchors on-the-fly, uses node distance thresholding
void Warp3dPoints(
		open3d::core::Tensor& warped_vertices, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
) {

	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPoints<o3c::Device::DeviceType::CPU>(
						warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage, minimum_valid_anchor_count,
						extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPoints<o3c::Device::DeviceType::CUDA>(
								warped_vertices, points, nodes, node_rotations, node_translations, anchor_count, node_coverage,
								minimum_valid_anchor_count,
								extrinsics
						);
				);
			}
	);
}

// uses provided precomputed anchors, does not use node distance thresholding
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPoints<o3c::Device::DeviceType::CPU>(
						warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights, extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPoints<o3c::Device::DeviceType::CUDA>(
								warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights, extrinsics
						);
				);
			}
	);
}

// uses provided precomputed anchors, uses node distance thresholding
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPoints<o3c::Device::DeviceType::CPU>(
						warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights, minimum_valid_anchor_count,
						extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPoints<o3c::Device::DeviceType::CUDA>(
								warped_points, points, nodes, node_rotations, node_translations, anchors, anchor_weights, minimum_valid_anchor_count,
								extrinsics
						);
				);
			}
	);
}

// endregion
// region =========================== POINTS AND NORMALS =============================================================================================
// computes anchors on-the-fly, does not use node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPointsAndNormals<o3c::Device::DeviceType::CPU>(
						warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchor_count, node_coverage,
						extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPointsAndNormals<o3c::Device::DeviceType::CUDA>(
								warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchor_count, node_coverage,
								extrinsics
						);
				);
			}
	);
}

// computes anchors on-the-fly, uses node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
) {

	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPointsAndNormals<o3c::Device::DeviceType::CPU>(
						warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchor_count, node_coverage,
						minimum_valid_anchor_count, extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPointsAndNormals<o3c::Device::DeviceType::CUDA>(
								warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchor_count, node_coverage,
								minimum_valid_anchor_count, extrinsics
						);
				);
			}
	);
}

// uses provided precomputed anchors, does not use node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPointsAndNormals<o3c::Device::DeviceType::CPU>(
						warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchors, anchor_weights,
						extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPointsAndNormals<o3c::Device::DeviceType::CUDA>(
								warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchors, anchor_weights,
								extrinsics
						);
				);
			}
	);
}

// uses provided precomputed anchors, uses node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
) {
	core::InferDeviceFromEntityAndExecute(
			nodes,
			[&] {
				Warp3dPointsAndNormals<o3c::Device::DeviceType::CPU>(
						warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchors, anchor_weights,
						minimum_valid_anchor_count, extrinsics
				);
			},
			[&] {
				NNRT_IF_CUDA(
						Warp3dPointsAndNormals<o3c::Device::DeviceType::CUDA>(
								warped_points, warped_normals, points, normals, nodes, node_rotations, node_translations, anchors, anchor_weights,
								minimum_valid_anchor_count, extrinsics
						);
				);
			}
	);
}
// endregion
} // namespace nnrt::geometry::kernel::warp