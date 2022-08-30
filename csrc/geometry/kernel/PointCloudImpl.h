//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 8/29/22.
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

//3rd-party
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/t/geometry/kernel/Transform.h>
#include <open3d/t/geometry/Utility.h>
#include <open3d/core/Dispatch.h>
#include <open3d/core/ParallelFor.h>

//local
#include "geometry/kernel/PointCloud.h"
#include "core/PlatformIndependentAtomics.h"


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;

namespace nnrt::geometry::kernel::pointcloud {

template<typename TDataPointer>
inline void InitializePointAttributeTensor(o3c::Tensor& output_attribute, TDataPointer** attribute_to_index, int64_t row_count,
                                           int64_t column_count, int64_t channel_count, const o3c::Dtype& dtype,
                                           const o3c::Device& device, bool preserve_image_layout) {
	if (channel_count == 1) {
		output_attribute = preserve_image_layout ?
		                   o3c::Tensor::Zeros({row_count, column_count}, dtype, device) :
		                   o3c::Tensor::Zeros({row_count * column_count}, dtype, device);
	} else {
		output_attribute = preserve_image_layout ?
		                   o3c::Tensor::Zeros({row_count, column_count, channel_count}, dtype, device) :
		                   o3c::Tensor::Zeros({row_count * column_count, channel_count}, dtype, device);
	}
	*attribute_to_index = output_attribute.template GetDataPtr<TDataPointer>();
}

template<o3c::Device::DeviceType TDeviceType, typename TDepth>
static void UnprojectWithoutDepthFiltering_TypeDispatched(
		open3d::core::Tensor& points, open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> colors,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> mask,
		const open3d::core::Tensor& depth, open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> image_colors,
		const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
) {
	if (image_colors.has_value() != colors.has_value()) {
		utility::LogError(
				"[Unproject] Both or none of image_colors and colors must have "
				"values.");
	}

	const o3c::Device device = depth.GetDevice();

	const bool have_colors = image_colors.has_value();
	if (have_colors) {
		o3c::AssertTensorDevice(image_colors.value(), device);
	}
	const bool build_mask = mask.has_value();

	o3tgk::NDArrayIndexer depth_indexer(depth, 2);
	o3tgk::NDArrayIndexer image_colors_indexer;

	o3c::Tensor pose = open3d::t::geometry::InverseTransformation(extrinsics);
	o3tgk::TransformIndexer transform(intrinsics, pose, 1.0f);

	int64_t rows = depth_indexer.GetShape(0);
	int64_t cols = depth_indexer.GetShape(1);

	float* point_data;
	InitializePointAttributeTensor(points, &point_data, rows, cols, 3, o3c::Float32, device, preserve_image_layout);

	float* color_data;
	if (have_colors) {
		const auto& image_color = image_colors.value().get();
		image_colors_indexer = o3tgk::NDArrayIndexer{image_color, 2};
		InitializePointAttributeTensor(colors.value().get(), &color_data, rows, cols, 3, o3c::Float32, device, preserve_image_layout);
	}

	bool* mask_data;
	if (build_mask) {
		InitializePointAttributeTensor(mask.value().get(), &mask_data, rows, cols, 1, o3c::Bool, device, preserve_image_layout);
	}

	int64_t point_count = rows * cols;

	o3c::ParallelFor(
			device, point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				int64_t y = workload_idx / cols;
				int64_t x = workload_idx % cols;

				float depth_metric = *depth_indexer.GetDataPtr<TDepth>(x, y) / depth_scale;
				auto* vertex = point_data + workload_idx * 3;
				if (depth_metric > 0 && depth_metric < depth_max) {
					float x_camera = 0, y_camera = 0, z_camera = 0;
					transform.Unproject(static_cast<float>(x), static_cast<float>(y), depth_metric, &x_camera, &y_camera, &z_camera);
					transform.RigidTransform(x_camera, y_camera, z_camera, vertex + 0, vertex + 1, vertex + 2);
					if (have_colors) {
						auto* pcd_pixel = color_data + workload_idx * 3;
						auto* image_pixel = image_colors_indexer.GetDataPtr<float>(x, y);
						*pcd_pixel = *image_pixel;
						*(pcd_pixel + 1) = *(image_pixel + 1);
						*(pcd_pixel + 2) = *(image_pixel + 2);
					}
					if (build_mask) {
						auto mask_ptr = mask_data + workload_idx;
						*mask_ptr = true;
					}
				}
			} /*end lambda*/
	); // end o3c::ParallelFor call
}


template<o3c::Device::DeviceType TDeviceType>
void UnprojectWithoutDepthFiltering(
		open3d::core::Tensor& points, open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> colors,
		open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> mask,
		const open3d::core::Tensor& depth, open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> image_colors,
		const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
) {
	DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
		UnprojectWithoutDepthFiltering_TypeDispatched<TDeviceType, scalar_t>(
				points, colors, mask, depth, image_colors, intrinsics, extrinsics,
				depth_scale, depth_max, preserve_image_layout
		);
	} /*end lambda*/); // end macro call DISPATCH_DTYPE_TO_TEMPLATE
}

}//namespace nnrt::geometry::kernel::pointcloud