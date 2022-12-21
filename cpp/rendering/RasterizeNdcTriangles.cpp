//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/5/22.
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
// 3rd party
#include <open3d/utility/Logging.h>
#include <open3d/t/geometry/Utility.h>

// local
#include "rendering/RasterizeNdcTriangles.h"
#include "rendering/kernel/RasterizeNdcTriangles.h"
#include "rendering/kernel/RasterizationConstants.h"
#include "rendering/kernel/CoordinateSystemConversions.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tg = open3d::t::geometry;

namespace nnrt::rendering {

std::tuple<open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor, open3d::core::Tensor>
RasterizeNdcTriangles(
		const open3d::core::Tensor& ndc_face_vertices,
		open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> clipped_faces_mask,
		const open3d::core::SizeVector& image_size,
		float blur_radius_pixels,
		int faces_per_pixel,
		int bin_size/* = -1*/,
		int max_faces_per_bin/* = -1*/,
		bool perspective_correct_barycentric_coordinates /* = false */,
		bool clip_barycentric_coordinates /* = false */,
		bool cull_back_faces /* = true */
) {
	if (faces_per_pixel > MAX_POINTS_PER_PIXEL) {
		utility::LogError("Need faces_per_pixel <= {}. Got: {}", MAX_POINTS_PER_PIXEL, faces_per_pixel);
	}
	if (image_size.size() != 2) {
		utility::LogError("image_size should be a SizeVector of size 2. Got size {}.", image_size.size());
	}
	int64_t max_image_dimension = std::max(image_size[0], image_size[1]);

	if (bin_size == -1) {
		if (max_image_dimension <= 64) {
			bin_size = 8;
		} else {
			/*
			 * Heuristic based formula maps max_image_dimension to bin_size as follows:
			 * max_image_dimension < 64 -> 8
			 * 16 < max_image_dimension < 256 -> 16
			 * 256 < max_image_dimension < 512 -> 32
			 * 512 < max_image_dimension < 1024 -> 64
			 * 1024 < max_image_dimension < 2048 -> 128
			 */
			bin_size =
					static_cast<int>(std::pow(2, std::max(static_cast<int>(std::ceil(std::log2(static_cast<double>(max_image_dimension)))) - 4, 4)));
		}
	}

	if (bin_size != 0) {
		int bins_along_max_dimension = 1 + (max_image_dimension - 1) / bin_size;
		if (bins_along_max_dimension >= MAX_BINS_ALONG_IMAGE_DIMENSION) {
			utility::LogError("The provided bin_size is too small: computed bin count along maximum dimension is {}, this has to be < {}.",
			                  bins_along_max_dimension, MAX_BINS_ALONG_IMAGE_DIMENSION);
		}
	}

	if (max_faces_per_bin == -1) {
		if (clipped_faces_mask.has_value()) {
			int32_t unclipped_face_count = static_cast<int32_t>(clipped_faces_mask.value().get().NonZero().GetShape(1));
			max_faces_per_bin = std::max(10000, unclipped_face_count / 5);
		} else {
			max_faces_per_bin = std::max(10000, static_cast<int>(ndc_face_vertices.GetLength()) / 5);
		}
	}

	o3c::AssertTensorDtype(ndc_face_vertices, o3c::Float32);


	kernel::Fragments fragments;
    float blur_radius_ndc = kernel::ImageSpaceDistanceToNdc(blur_radius_pixels, image_size[0], image_size[1]);

	if (bin_size > 0 && max_faces_per_bin > 0) {
		// Use coarse-to-fine rasterization
		o3c::Tensor bin_faces;
        kernel::GridBinNdcTriangles(bin_faces,
                                    ndc_face_vertices,
                                    clipped_faces_mask,
                                    image_size,
                                    blur_radius_ndc,
                                    bin_size,
                                    max_faces_per_bin);

        kernel::RasterizeNdcTriangles_GridBinned(fragments,
                                                 ndc_face_vertices,
                                                 bin_faces,
                                                 image_size,
                                                 blur_radius_ndc,
                                                 bin_size,
                                                 faces_per_pixel,
                                                 perspective_correct_barycentric_coordinates,
                                                 clip_barycentric_coordinates,
                                                 cull_back_faces);
	} else {

		// Use the naive per-pixel implementation
        kernel::RasterizeNdcTriangles_BruteForce(fragments,
                                                 ndc_face_vertices,
                                                 clipped_faces_mask,
                                                 image_size,
                                                 blur_radius_ndc,
                                                 faces_per_pixel,
                                                 perspective_correct_barycentric_coordinates,
                                                 clip_barycentric_coordinates,
                                                 cull_back_faces);

	}
	return fragments.ToTuple();
}


} // namespace nnrt::rendering