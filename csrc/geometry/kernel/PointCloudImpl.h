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

inline void InitializePointAttributeTensors(o3c::Tensor& output_attribute, o3c::Tensor& attribute_to_index, int64_t row_count,
                                     int64_t column_count, int64_t channel_count, const o3c::Device& device, bool preserve_image_layout){
	if (preserve_image_layout) {
		output_attribute = o3c::Tensor({row_count, column_count, channel_count}, o3c::Float32, device);
		attribute_to_index = output_attribute.View({row_count * column_count, channel_count});
	} else {
		output_attribute = o3c::Tensor({row_count * column_count, channel_count}, o3c::Float32, device);
		attribute_to_index = output_attribute;
	}
}

template<o3c::Device::DeviceType TDeviceType>
void UnprojectWithoutDepthFiltering(
		o3c::Tensor& points, utility::optional <std::reference_wrapper<o3c::Tensor>> colors,
		utility::optional <std::reference_wrapper<o3c::Tensor>> mask,
		const o3c::Tensor& depth, utility::optional <std::reference_wrapper<const o3c::Tensor>> image_colors,
		const o3c::Tensor& intrinsics, const o3c::Tensor& extrinsics, float depth_scale, float depth_max, bool preserve_image_layout
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

	o3c::Tensor points_to_index;
	InitializePointAttributeTensors(points, points_to_index, rows, cols, 3, device, preserve_image_layout);

	o3tgk::NDArrayIndexer point_indexer(points_to_index, 1);
	o3c::Tensor colors_to_index;
	o3tgk::NDArrayIndexer colors_indexer;
	if (have_colors) {
		const auto& image_color = image_colors.value().get();
		image_colors_indexer = o3tgk::NDArrayIndexer{image_color, 2};
		InitializePointAttributeTensors(colors.value().get(), colors_to_index, rows, cols, 3, device, preserve_image_layout);
		colors_indexer = o3tgk::NDArrayIndexer(colors_to_index, 1);
	}
	o3c::Tensor masks_to_index;
	o3tgk::NDArrayIndexer mask_indexer;
	if (build_mask) {
		InitializePointAttributeTensors(mask.value().get(), masks_to_index, rows, cols, 1, device, preserve_image_layout);
		mask_indexer = o3tgk::NDArrayIndexer(mask.value().get(), 1);
	}

	int64_t point_count = rows * cols;

	DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
		o3c::ParallelFor(device, point_count,
		                 [=]
		OPEN3D_DEVICE(int64_t
		workload_idx) {
		int64_t y = workload_idx / cols;
		int64_t x = workload_idx % cols;

		float depth_metric = *depth_indexer.GetDataPtr<scalar_t>(x, y) / depth_scale;
		auto* vertex = point_indexer.GetDataPtr<float>(workload_idx);
		if (depth_metric > 0 && depth_metric < depth_max) {
			float x_camera = 0, y_camera = 0, z_camera = 0;
			transform.Unproject(static_cast<float>(x), static_cast<float>(y), depth_metric, &x_camera, &y_camera, &z_camera);
			transform.RigidTransform(x_camera, y_camera, z_camera, vertex + 0, vertex + 1, vertex + 2);
			if (have_colors) {
				auto* pcd_pixel = colors_indexer.GetDataPtr<float>(workload_idx);
				auto* image_pixel = image_colors_indexer.GetDataPtr<float>(x, y);
				*pcd_pixel = *image_pixel;
				*(pcd_pixel + 1) = *(image_pixel + 1);
				*(pcd_pixel + 2) = *(image_pixel + 2);
			}
			if (build_mask) {
				auto mask_ptr = mask_indexer.GetDataPtr<bool>(workload_idx);
				*mask_ptr = true;
			}
		} else {
			*vertex = 0.f;
			*(vertex + 1) = 0.f;
			*(vertex + 2) = 0.f;
		}
	} /*end lambda*/ ); // end o3c::ParallelFor call
	} /*end lambda*/); // end macro call DISPATCH_DTYPE_TO_TEMPLATE
}

}//namespace nnrt::geometry::kernel::pointcloud