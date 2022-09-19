//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/6/22.
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

#include <open3d/core/Tensor.h>
#include "core/PlatformIndependence.h"
#include "AxisAlignedBoundingBox.h"
#include <open3d/t/geometry/Utility.h>

namespace nnrt::rendering::kernel {

// NOTE: We define normalized camera space as normalized to the frustum in x/y plane, i.e. (-1,-1) to (1, 1).

/*
 * The default value of the normalized camera-space range is [-1, 1], however in
 * the case that image_height != image_width, the range is set such that the
 * shorter side has range [-1, 1] and the longer side is scaled by the ratio of
 * image_height:image_width.
 * To get the normalized camera-space range in the x direction,
 * dimension1 = image_width and dimension2 = image_height.
 */
template<typename TDimension>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline float GetNormalizedCameraSpaceRange(TDimension dimension1, TDimension dimension2) {
	float range = 2.0f;
	if (dimension1 > dimension2) {
		range = (static_cast<float>(dimension1) * range) / static_cast<float>(dimension2);
	}
	return range;
}

/*
 * Given a pixel coordinate 0 <= i < S1, convert it to normalized camera-space coordinates.
 * We divide the normalized camera-space range into S1 evenly-sized pixels, and assume that
 * each pixel falls in the *center* of its range.
 * The default normalized camera range is [-1, 1]. However, in the case
 * that image_height != image_width, the range is set such that the shorter side
 * has range [-1, 1] and the longer side is scaled by the image_height:image_width
 * ratio. For example, to get the x and y normalized camera-space coordinates for a given
 * pixel (u,v), proceed as follows:
 *     x = ImageToScreenSpace(u, image_width, image_height)
 *     y = ImageToScreenSpace(v, image_height, image_width)
 */
NNRT_DEVICE_WHEN_CUDACC
inline float ImageSpaceToNormalizedCameraSpace(int i_pixel_along_dimension1, int dimension1, int dimension2) {
	float range = GetNormalizedCameraSpaceRange(dimension1, dimension2);
	const float offset = (range / 2.0f);
	return -offset + (range * static_cast<float>(i_pixel_along_dimension1) + offset) / static_cast<float>(dimension1);
}

// Candidate function to be moved into a separate "camera" module
inline
std::tuple<open3d::core::Tensor, kernel::AxisAligned2dBoundingBox>
IntrinsicsToNormalizedCameraSpaceAndRange(const open3d::core::Tensor& intrinsics, const open3d::core::SizeVector& image_size) {
	open3d::t::geometry::CheckIntrinsicTensor(intrinsics);
	auto values = intrinsics.ToFlatVector<double>();
	double fx = values[0];
	double fy = values[4];
	double cx = values[2];
	double cy = values[5];

	auto height = static_cast<double>(image_size[0]);
	auto width = static_cast<double>(image_size[1]);

	double smaller_dimension = std::min(width, height);
	float range_x = GetNormalizedCameraSpaceRange(static_cast<int>(image_size[1]), static_cast<int>(image_size[0]));
	float range_y = GetNormalizedCameraSpaceRange(static_cast<int>(image_size[0]), static_cast<int>(image_size[1]));

	/*
	 * The default normalized camera range is [-1, 1]. However, in the case
	 * that image_height != image_width, the range is set such that the shorter side
	 * has range [-1, 1] and the longer side is scaled by the image_height:image_width
	 * ratio.
	 */
	double fx_normalized = 2.0 * fx / smaller_dimension;
	// important: fy is negated to flip the y-axis from going up to going down, as in an image (or on screen)
	double fy_normalized = -2.0 * fy / smaller_dimension;
	double cx_normalized = -(2.0 * cx - width) / smaller_dimension;
	// likewise, we (don't) negate cy_normalized here in order to flip top<->bottom and transition to screen space convention
	double cy_normalized = (2.0 * cy - height) / smaller_dimension;
	open3d::core::Tensor normalized_camera_intrinsic_matrix(std::vector<double>{fx_normalized, 0.0, cx_normalized,
	                                                                            0.0, fy_normalized, cy_normalized,
	                                                                            0.0, 0.0, 1.0}, {3, 3}, open3d::core::Float64,
	                                                        open3d::core::Device("CPU:0"));
	kernel::AxisAligned2dBoundingBox range{static_cast<float>(cx_normalized - range_x / 2.f),
	                                                 static_cast<float>(cx_normalized + range_x / 2.f),
	                                                 static_cast<float>(cy_normalized - range_y / 2.f),
	                                                 static_cast<float>(cy_normalized + range_y / 2.f)};

	return std::make_tuple(normalized_camera_intrinsic_matrix, range);
}

} // namespace nnrt::rendering::kernel