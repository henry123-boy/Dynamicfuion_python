//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 12/12/22.
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
// stdlib includes

// third-party includes
#include <open3d/core/ParallelFor.h>

// local includes
#include "rendering/kernel/FlatEdgeShader.h"
#include "core/PlatformIndependence.h"
#include "core/PlatformIndependentArray.h"

namespace utility = open3d::utility;
namespace o3c = open3d::core;

namespace nnrt::rendering::kernel {
template<open3d::core::Device::DeviceType TDeviceType>
void ShadeEdgesFlat(
        open3d::core::Tensor& pixels,
        const open3d::core::Tensor& pixel_face_indices,
        const open3d::core::Tensor& pixel_depths,
        const open3d::core::Tensor& pixel_barycentric_coordinates,
        const open3d::core::Tensor& pixel_face_distances,
        open3d::utility::optional<std::reference_wrapper<const std::vector<open3d::t::geometry::TriangleMesh>>> meshes,
        float ndc_line_width,
        const std::array<float, 3>& color
) {
    o3c::Device device = pixel_face_indices.GetDevice();

    if (ndc_line_width <= 0.f) {
        utility::LogError("Expecting a non-zero, positive ndc_line_width. Got: {}.", ndc_line_width);
    }
    int64_t image_height = pixel_face_indices.GetShape(0);
    int64_t image_width = pixel_face_indices.GetShape(1);
    pixels = o3c::Tensor::Zeros({image_height, image_width, 3}, o3c::UInt8, device);
    auto pixels_data = pixels.GetDataPtr<u_char>();

    int64_t pixel_count = image_height * image_width;

    float ndc_line_width_squared = ndc_line_width * ndc_line_width;
//    float quarter_ndc_line_width_squared = ndc_line_width_squared / 4.0f;
//    float three_quarter_ndc_line_width_squared = 3.f * quarter_ndc_line_width_squared;

    int64_t face_count_per_pixel = pixel_face_indices.GetShape(2);
    auto pixel_face_distance_data = pixel_face_distances.GetDataPtr<float>();
    auto pixel_face_index_data = pixel_face_indices.GetDataPtr<int64_t>();

    float r = color[0];
    float g = color[1];
    float b = color[2];

    o3c::ParallelFor(
            device, pixel_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
                int64_t v_pixel = workload_idx / image_width;
                int64_t u_pixel = workload_idx % image_width;
                int64_t fragment_position =
                        (v_pixel * image_width * face_count_per_pixel) + (u_pixel * face_count_per_pixel);
                int64_t pixel_face_index = pixel_face_index_data[fragment_position];
                if(pixel_face_index == -1){
                    return;
                }
                float pixel_face_distance = fabsf(pixel_face_distance_data[fragment_position]);

                if (pixel_face_distance < ndc_line_width_squared) {
                    auto pixel_memory_location = v_pixel * image_width * 3 + u_pixel * 3;
//                    if (pixel_face_distance < quarter_ndc_line_width_squared) {
//                        pixels_data[pixel_memory_location + 0] = static_cast<u_char>(255.f * r);
//                        pixels_data[pixel_memory_location + 1] = static_cast<u_char>(255.f * g);
//                        pixels_data[pixel_memory_location + 2] = static_cast<u_char>(255.f * b);
//                    } else {
//                        float factor =
//                                (three_quarter_ndc_line_width_squared - pixel_face_distance)
//                                / three_quarter_ndc_line_width_squared;
                        float factor = (ndc_line_width_squared-pixel_face_distance) / ndc_line_width_squared;
                        pixels_data[pixel_memory_location + 0] = static_cast<u_char>(255.f * r * factor);
                        pixels_data[pixel_memory_location + 1] = static_cast<u_char>(255.f * g * factor);
                        pixels_data[pixel_memory_location + 2] = static_cast<u_char>(255.f * b * factor);
//                    }
                }

            }
    );

}

} // namespace nnrt::rendering::kernel