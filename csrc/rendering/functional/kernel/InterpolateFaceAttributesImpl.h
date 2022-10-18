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
#pragma once

//3rd party
#include <open3d/core/Dispatch.h>
#include <open3d/core/ParallelFor.h>

//local
#include "rendering/functional/kernel/InterpolateFaceAttributes.h"
#include "core/PlatformIndependence.h"

namespace o3c = open3d::core;
namespace utility = open3d::utility;

typedef int32_t t_face_index;

namespace nnrt::rendering::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType, typename TAttribute>
void InterpolateFaceAttributes_Dispatched(open3d::core::Tensor& pixel_attributes, const open3d::core::Tensor& pixel_face_indices,
                                          const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& face_attributes) {
	o3c::Device device = face_attributes.GetDevice();
	o3c::Dtype attribute_dtype = face_attributes.GetDtype();
	int64_t attribute_element_count = face_attributes.GetShape(2);
	int64_t pixel_count = pixel_face_indices.GetShape(0);
	int64_t per_pixel_face_count = pixel_face_indices.GetShape(1);

	const auto* pixel_face_index_ptr = pixel_face_indices.template GetDataPtr<t_face_index>();
	const auto* pixel_barycentric_coordinates_ptr = pixel_barycentric_coordinates.template GetDataPtr<float>();
	const auto* face_attribute_ptr = face_attributes.template GetDataPtr<TAttribute>();

	pixel_attributes = o3c::Tensor::Zeros({pixel_count, per_pixel_face_count, attribute_element_count}, attribute_dtype, device);

	auto* pixel_attribute_ptr = pixel_attributes.template GetDataPtr<TAttribute>();

	o3c::ParallelFor(
			device, pixel_attributes.NumElements(),
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const int64_t i_pixel = workload_idx / attribute_element_count;
				const int64_t i_attribute_element = workload_idx % attribute_element_count;

				for (int i_pixel_face = 0; i_pixel_face < per_pixel_face_count; i_pixel_face++) {
					const t_face_index i_face = pixel_face_index_ptr[i_pixel * per_pixel_face_count + i_pixel_face];
					if (i_face < 0) {
						break; // assume -1 is sentinel face_index value, after which there are no more non-negative entries
					}
					TAttribute pixel_attribute = 0.0;
					for (int i_triangle_vertex = 0; i_triangle_vertex < 3; i_triangle_vertex++) {
						float weight = pixel_barycentric_coordinates_ptr[(i_pixel * per_pixel_face_count + i_pixel_face) * 3 + i_triangle_vertex];
						TAttribute vertex_attribute = face_attribute_ptr[i_face * 3 * attribute_element_count +
						                                                 i_triangle_vertex * attribute_element_count + i_attribute_element];
						pixel_attribute += weight * vertex_attribute;
					}
					pixel_attribute_ptr[i_pixel * per_pixel_face_count * attribute_element_count +
					                    i_pixel_face * attribute_element_count + i_attribute_element] = pixel_attribute;
				}
			}
	);
}


template<open3d::core::Device::DeviceType TDeviceType>
void InterpolateFaceAttributes(open3d::core::Tensor& interpolated_attributes, const open3d::core::Tensor& pixel_face_indices,
                               const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& face_attributes) {

	DISPATCH_DTYPE_TO_TEMPLATE(face_attributes.GetDtype(), [&]() {
		InterpolateFaceAttributes_Dispatched<TDeviceType, scalar_t>(
				interpolated_attributes, pixel_face_indices,
				pixel_barycentric_coordinates, face_attributes
		);
	});
}

} // namespace nnrt::rendering::functional::kernel