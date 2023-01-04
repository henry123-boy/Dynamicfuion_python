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

// TODO Candidate functions to be moved into a separate "camera" module
// ==== KERNEL-LEVEL FUNCTIONS =====

// 3rd party
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/Utility.h>

// local
#include "core/PlatformIndependentQualifiers.h"
#include "core/PlatformIndependentAlgorithm.h"
#include "geometry/kernel/AxisAlignedBoundingBox.h"

namespace nnrt::rendering::kernel {

typedef geometry::kernel::AxisAligned2dBoundingBox AxisAligned2dBoundingBox;

// NOTE: We define normalized camera space as normalized to the frustum in x/y plane, i.e. (-1,-1) to (1, 1).

/*
 * The NDC range is set such that the shorter image side has range [-1, 1] and the longer side range is scaled up by
 * the longer_side:shorter_side ratio, i.e. if image_width > image_height, the NDC range along the x-axis is
 * -image_width/image_height, image_width/image_height.
 *
 * To get the normalized camera-space range in the x direction,
 * dimension1 = image_width and dimension2 = image_height.
 */
template<typename TPixelCount>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline float GetNdcRange(const TPixelCount dimension1,const TPixelCount dimension2) {
	float range = 2.0f;
	if (dimension1 > dimension2) {
		range = (static_cast<float>(dimension1) * range) / static_cast<float>(dimension2);
	}
	return range;
}

/*
 * The NDC range is set such that the shorter image side has range [-1, 1] and the longer side range is scaled up by
 * the longer_side:shorter_side ratio, i.e. if image_width > image_height, the NDC range along the x-axis is
 * -image_width/image_height, image_width/image_height.
 *
 * To get the NDC x and y for a given pixel (u,v), proceed as follows:
 *     x_ndc = ImageSpacePixelAlongDimensionToNdc(u, image_width, image_height)
 *     y_ndc = ImageSpacePixelAlongDimensionToNdc(v, image_height, image_width)
 */
template<typename TPixelCount>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline float ImageSpacePixelAlongDimensionToNdc(
        const TPixelCount i_pixel_along_dimension1, const TPixelCount dimension1, const TPixelCount dimension2){
	float range = GetNdcRange(dimension1, dimension2);
	const float offset = (range / 2.0f);
	return -offset + (range * static_cast<float>(i_pixel_along_dimension1) + offset) / static_cast<float>(dimension1);
}

template<typename  TPixelCount>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline void ImageSpacePixelToNdc(float& x_ndc, float& y_ndc, const TPixelCount u_pixel, const TPixelCount v_pixel,
                                 const TPixelCount image_width, const TPixelCount image_height) {
	x_ndc = ImageSpacePixelAlongDimensionToNdc(u_pixel, image_width, image_height);
    y_ndc = ImageSpacePixelAlongDimensionToNdc(v_pixel, image_height, image_width);
}

template<typename TPixelCount>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline TPixelCount NdcAlongDimensionToImageSpacePixel(
        const float ndc_along_dimension1, const TPixelCount dimension1, const TPixelCount dimension2){
	float range = GetNdcRange(dimension1, dimension2);
	const float offset = (range / 2.0f);
	return static_cast<TPixelCount>((static_cast<float>(dimension1) * (ndc_along_dimension1 + offset) - offset) / range);
}

template<typename  TPixelCount>
NNRT_HOST_DEVICE_WHEN_CUDACC
inline void NdcToImageSpace(TPixelCount& u_pixel, TPixelCount& v_pixel, const float x_ndc, const float y_ndc,
                             const int image_width, const int image_height) {
	u_pixel = NdcAlongDimensionToImageSpacePixel(x_ndc, image_width, image_height);
    v_pixel = NdcAlongDimensionToImageSpacePixel(y_ndc, image_height, image_width);
}

NNRT_HOST_DEVICE_WHEN_CUDACC
inline float ImageSpaceDistanceToNdc(float distance_pixels, int dimension1, int dimension2) {
	auto half_min_dim = static_cast<float>(fminf(dimension1, dimension2)) / 2.0f;
	return distance_pixels / half_min_dim;
}

//TODO: move host-level code somewhere it makes more sense. Perhaps reorganize -- things having to do with projections
// need to go to the geometry namespace (or, perhaps, a geometry::projection sub-space).

// ==== HOST-LEVEL FUNCTIONS =====
inline
std::tuple<open3d::core::Tensor, kernel::AxisAligned2dBoundingBox>
ImageSpaceIntrinsicsToNdc(const open3d::core::Tensor& intrinsics, const open3d::core::SizeVector& image_size) {
	open3d::t::geometry::CheckIntrinsicTensor(intrinsics);
	auto values = intrinsics.ToFlatVector<double>();
	double fx = values[0];
	double fy = values[4];
	double cx = values[2];
	double cy = values[5];

	auto height = static_cast<double>(image_size[0]);
	auto width = static_cast<double>(image_size[1]);

	double smaller_dimension = std::min(width, height);
	float range_x = GetNdcRange(static_cast<int>(image_size[1]), static_cast<int>(image_size[0]));
	float range_y = GetNdcRange(static_cast<int>(image_size[0]), static_cast<int>(image_size[1]));

	/*
	 * The default normalized camera range is [-1, 1]. However, in the case
	 * that image_height != image_width, the range is set such that the shorter side
	 * has range [-1, 1] and the longer side is scaled by the image_height:image_width
	 * ratio.
	 */
	double fx_ndc = 2.0 * fx / smaller_dimension;
	// important: fy is negated to flip the y-axis from going up to going down, as in an image (or on screen)
	double fy_ndc = -2.0 * fy / smaller_dimension;
	double cx_ndc = -(2.0 * cx - width) / smaller_dimension;
	// likewise, we (don't) negate cy_normalized here in order to flip top<->bottom and transition to screen space convention
	double cy_ndc = (2.0 * cy - height) / smaller_dimension;
	open3d::core::Tensor normalized_camera_intrinsic_matrix(std::vector<double>{fx_ndc, 0.0, cx_ndc,
                                                                                0.0, fy_ndc, cy_ndc,
                                                                                0.0, 0.0, 1.0}, {3, 3}, open3d::core::Float64,
	                                                        open3d::core::Device("CPU:0"));
	kernel::AxisAligned2dBoundingBox range{static_cast<float>(cx_ndc - range_x / 2.f),
	                                                 static_cast<float>(cx_ndc + range_x / 2.f),
	                                                 static_cast<float>(cy_ndc - range_y / 2.f),
	                                                 static_cast<float>(cy_ndc + range_y / 2.f)};

	return std::make_tuple(normalized_camera_intrinsic_matrix, range);
}

} // namespace nnrt::rendering::kernel