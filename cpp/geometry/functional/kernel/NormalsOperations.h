//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 7/21/22.
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

namespace nnrt::geometry::kernel::mesh {

void ComputeTriangleNormals(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                            const open3d::core::Tensor& triangle_indices);

template<open3d::core::Device::DeviceType TDevice>
void ComputeTriangleNormals(open3d::core::Tensor& triangle_normals, const open3d::core::Tensor& vertex_positions,
                            const open3d::core::Tensor& triangle_indices);

void NormalizeVectors3d(open3d::core::Tensor& vectors3f);

template<open3d::core::Device::DeviceType TDevice>
void NormalizeVectors3d(open3d::core::Tensor& vectors3f);

void ComputeVertexNormals(open3d::core::Tensor& vertex_normals, const open3d::core::Tensor& triangle_indices,
                          const open3d::core::Tensor& triangle_normals);

template<open3d::core::Device::DeviceType TDevice>
void ComputeVertexNormals(open3d::core::Tensor& vertex_normals, const open3d::core::Tensor& triangle_indices,
                          const open3d::core::Tensor& triangle_normals);
} // nnrt::geometry::kernel::mesh

namespace nnrt::geometry::kernel::point_cloud {
void ComputeOrderedPointCloudNormals(open3d::core::Tensor& normals, const open3d::core::Tensor& point_positions,
									 const open3d::core::SizeVector& source_image_size);
template<open3d::core::Device::DeviceType TDevice>
void ComputeOrderedPointCloudNormals(open3d::core::Tensor& normals, const open3d::core::Tensor& point_positions,
                                     const open3d::core::SizeVector& source_image_size);

} // nnrt::geometry::kernel::point_cloud