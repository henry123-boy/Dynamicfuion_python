//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 10/18/22.
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
// local
#include "rendering/functional/InterpolateVertexAttributes.h"
#include "rendering/functional/kernel/InterpolateFaceAttributes.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

namespace nnrt::rendering::functional {

open3d::core::Tensor InterpolateVertexAttributes(
		const open3d::core::Tensor& pixel_face_indices,
		const open3d::core::Tensor& pixel_barycentric_coordinates,
		const open3d::core::Tensor& face_attributes
) {
	o3c::AssertTensorShape(face_attributes, { utility::nullopt, 3 , utility::nullopt});
	o3c::AssertTensorDtypes(face_attributes, { o3c::Float32, o3c::Float64 });
	auto device = face_attributes.GetDevice();

	o3c::AssertTensorDevice(pixel_face_indices, device);
	o3c::AssertTensorDtype(pixel_face_indices, o3c::Int64);
	int64_t image_height = pixel_face_indices.GetShape(0);
	int64_t image_width = pixel_face_indices.GetShape(1);
	int64_t per_pixel_face_count = pixel_face_indices.GetShape(2);
	o3c::AssertTensorShape(pixel_face_indices, {image_height, image_width, per_pixel_face_count});

	o3c::AssertTensorDevice(pixel_barycentric_coordinates, device);
	o3c::AssertTensorDtype(pixel_barycentric_coordinates, o3c::Float32);
	o3c::AssertTensorShape(pixel_barycentric_coordinates, {image_height, image_width, per_pixel_face_count, 3});

	o3c::Tensor interpolated_attributes;
    kernel::InterpolateVertexAttributes(interpolated_attributes, pixel_face_indices, pixel_barycentric_coordinates,
                                        face_attributes);

	return interpolated_attributes;
}

} // namespace nnrt::rendering::functional