//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 9/19/22.
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

// 3rd-party
#include <open3d/core/Tensor.h>
#include <open3d/core/ParallelFor.h>

namespace nnrt::rendering::kernel {

template<>
void GridBin2dBoundingBoxes_Device<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& bins,
		const open3d::core::Tensor& bounding_boxes,
		const open3d::core::Tensor& boxes_to_skip_mask,
		const int image_height,
		const int image_width,
		const int grid_height_in_bins,
		const int grid_width_in_bins,
		const int bin_side_length,
		const int bin_capacity,
		const float half_pixel_x,
		const float half_pixel_y
) {
	auto bounding_box_count = bounding_boxes.GetShape(1);
	auto bin_data = bins.GetDataPtr<int32_t>();
	auto device = bounding_boxes.GetDevice();
	auto boxes_mask_data = boxes_to_skip_mask.GetDataPtr<bool>();
	auto bounding_box_data = bounding_boxes.GetDataPtr<float>();

	std::vector<std::atomic<int32_t>> bin_face_counts(grid_width_in_bins * grid_height_in_bins);
	// zero-out bin face counts
	o3c::ParallelFor(
			device, grid_width_in_bins * grid_height_in_bins,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				bin_face_counts[workload_idx].store(0);
			}
	);

	o3c::ParallelFor(
			device, bounding_box_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				if (boxes_mask_data[workload_idx]) {
					return;
				}
				float x_min = bounding_box_data[0 * bounding_box_count + workload_idx];
				float x_max = bounding_box_data[1 * bounding_box_count + workload_idx];
				float y_min = bounding_box_data[2 * bounding_box_count + workload_idx];
				float y_max = bounding_box_data[3 * bounding_box_count + workload_idx];

				for (int bin_y = 0; bin_y < grid_height_in_bins; bin_y++) {
					const float bin_y_min =
                            ImageSpacePixelAlongDimensionToNdc(bin_y * bin_side_length, image_height, image_width) - half_pixel_y;
					const float bin_y_max =
                            ImageSpacePixelAlongDimensionToNdc((bin_y + 1) * bin_side_length - 1, image_height,
                                                               image_width) + half_pixel_y;
					const bool y_overlap = (y_min <= bin_y_max) && (bin_y_min < y_max);

					if (y_overlap) {
						for (int bin_x = 0; bin_x < grid_width_in_bins; bin_x++) {
							const float bin_x_min =
                                    ImageSpacePixelAlongDimensionToNdc(bin_x * bin_side_length, image_width,
                                                                       image_height) - half_pixel_x;
							const float bin_x_max =
                                    ImageSpacePixelAlongDimensionToNdc((bin_x + 1) * bin_side_length - 1, image_width,
                                                                       image_height) + half_pixel_x;
							const bool x_overlap = (x_min <= bin_x_max) && (bin_x_min < x_max);

							if (x_overlap) {
								int32_t bin_index = bin_y * grid_width_in_bins + bin_x;
								int32_t insertion_position = bin_face_counts[bin_index].fetch_add(1);

								// store active face index into the bin if they overlap spatially
								bin_data[bin_index * bin_capacity + insertion_position] = static_cast<int32_t>(workload_idx);
							}
						}
					}
				}
			}
	);
}

} // namespace nnrt::rendering::kernel