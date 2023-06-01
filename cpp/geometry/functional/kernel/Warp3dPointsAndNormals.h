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
#pragma once

#include <open3d/core/Tensor.h>

#include "core/platform_independence/Qualifiers.h"
#include "Defines.h"
#include "geometry/functional/AnchorComputationMethod.h"

namespace nnrt::geometry::functional::kernel::warp {
// region ============== POINTS ONLY =================================================================================================================
// region ============== POINTS ONLY: FOR EXTERNAL USAGE =============================================================================================

// version not using node distance thresholding
void Warp3dPoints(open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
                  const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  int anchor_count, float node_coverage,
                  const open3d::core::Tensor& extrinsics);

// version using node distance thresholding
void Warp3dPoints(open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
                  const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  int anchor_count, float node_coverage, int minimum_valid_anchor_count,
                  const open3d::core::Tensor& extrinsics);

// version using precomputed anchors, but not distance thresholding
void Warp3dPoints(open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
                  const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
                  const open3d::core::Tensor& extrinsics);

// version using precomputed anchors & distance thresholding
void Warp3dPoints(open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
                  const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
                  const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
                  const open3d::core::Tensor& extrinsics);

// endregion
// region =========== POINTS ONLY: DEVICE-TEMPLATED IMPLEMENTATIONS OF THE ABOVE VERSIONS (IN SAME ORDER) ============================================

// computes anchors on-the-fly, does not use node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		const open3d::core::Tensor& extrinsics
);

// computes anchors on-the-fly, uses node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);

// uses provided precomputed anchors, does not use node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		const open3d::core::Tensor& extrinsics
);

// uses provided precomputed anchors, uses node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPoints(
		open3d::core::Tensor& warped_points, const open3d::core::Tensor& points,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);
// endregion
// endregion
// region ============== POINTS AND NORMALS ==========================================================================================================
// region ============== POINTS AND NORMALS: FOR EXTERNAL USE ========================================================================================

// version not using node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		const open3d::core::Tensor& extrinsics
);

// version using node distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);

// version using precomputed anchors, but not distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		const open3d::core::Tensor& extrinsics
);

// version using precomputed anchors & distance thresholding
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);
// endregion
// region ================== POINTS AND NORMALS: DEVICE-TEMPLATED IMPLEMENTATIONS OF THE SAME VERSIONS (IN SAME ORDER) ===============================

// computes anchors on-the-fly, does not use node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		const open3d::core::Tensor& extrinsics
);

// computes anchors on-the-fly, uses node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		int anchor_count, float node_coverage,
		int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);

// uses provided precomputed anchors, does not use node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights,
		const open3d::core::Tensor& extrinsics
);

// uses provided precomputed anchors, uses node distance thresholding
template<open3d::core::Device::DeviceType TDeviceType>
void Warp3dPointsAndNormals(
		open3d::core::Tensor& warped_points, open3d::core::Tensor& warped_normals,
		const open3d::core::Tensor& points, const open3d::core::Tensor& normals,
		const open3d::core::Tensor& nodes, const open3d::core::Tensor& node_rotations, const open3d::core::Tensor& node_translations,
		const open3d::core::Tensor& anchors, const open3d::core::Tensor& anchor_weights, int minimum_valid_anchor_count,
		const open3d::core::Tensor& extrinsics
);
// endregion
// endregion

} // namespace nnrt::geometry::functional::kernel::warp