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
#include "rendering/functional/kernel/InterpolateFaceAttributesImpl.h"

namespace nnrt::rendering::functional::kernel {
template void InterpolateVertexAttributes<open3d::core::Device::DeviceType::CPU>(
		open3d::core::Tensor& interpolated_attributes, const open3d::core::Tensor& pixel_face_indices,
		const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& face_attributes
);
} // namespace nnrt::rendering::functional::kernel