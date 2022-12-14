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
#include "geometry/functional/kernel/PerspectiveProjection.h"
#include "core/PlatformIndependentAtomics.h"
#include "core/PlatformIndependentQualifiers.h"

#ifdef __CUDACC__
#include <cuda/atomic>
#endif

//__DEBUG
#include <cuda/atomic>


namespace o3c = open3d::core;
namespace utility = open3d::utility;
namespace o3tgk = open3d::t::geometry::kernel;

namespace nnrt::geometry::functional::kernel {

template<typename TDataPointer>
inline void InitializePointAttributeTensor(
        o3c::Tensor& output_attribute, TDataPointer** attribute_to_index, int64_t row_count,
        int64_t column_count, int64_t channel_count, const o3c::Dtype& dtype,
        const o3c::Device& device, bool preserve_image_layout
) {
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
static void UnprojectRasterWithoutDepthFiltering_TypeDispatched(
        open3d::core::Tensor& points,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> colors,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> mask,
        const open3d::core::Tensor& depth,
        open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> image_colors,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        bool preserve_image_layout
) {
    o3c::AssertTensorDtypes(depth, { o3c::UInt16, o3c::Float32 });
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

    float* unprojected_point_data;
    InitializePointAttributeTensor(points, &unprojected_point_data, rows, cols, 3, o3c::Float32, device,
                                   preserve_image_layout);

    float* color_data;
    if (have_colors) {
        const auto& image_color = image_colors.value().get();
        image_colors_indexer = o3tgk::NDArrayIndexer{image_color, 2};
        InitializePointAttributeTensor(colors.value().get(), &color_data, rows, cols, 3, o3c::Float32, device,
                                       preserve_image_layout);
    }

    bool* mask_data;
    if (build_mask) {
        InitializePointAttributeTensor(mask.value().get(), &mask_data, rows, cols, 1, o3c::Bool, device,
                                       preserve_image_layout);
    }

    int64_t point_count = rows * cols;

    o3c::ParallelFor(
            device, point_count,
            [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float depth_metric = *depth_indexer.GetDataPtr<TDepth>(x, y) / depth_scale;
                auto* unprojected_point = unprojected_point_data + workload_idx * 3;
                if (depth_metric > 0 && depth_metric < depth_max) {
                    float x_camera = 0, y_camera = 0, z_camera = 0;
                    transform.Unproject(static_cast<float>(x), static_cast<float>(y), depth_metric, &x_camera,
                                        &y_camera, &z_camera);
                    transform.RigidTransform(x_camera, y_camera, z_camera, unprojected_point + 0,
                                             unprojected_point + 1, unprojected_point + 2);
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
void UnprojectRasterWithoutDepthFiltering(
        open3d::core::Tensor& unprojected_points,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> unprojected_point_colors,
        open3d::utility::optional<std::reference_wrapper<open3d::core::Tensor>> depth_mask,
        const open3d::core::Tensor& raster_depth,
        open3d::utility::optional<std::reference_wrapper<const open3d::core::Tensor>> raster_colors,
        const open3d::core::Tensor& intrinsics,
        const open3d::core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        bool preserve_image_layout
) {
    DISPATCH_DTYPE_TO_TEMPLATE(raster_depth.GetDtype(), [&]() {
        UnprojectRasterWithoutDepthFiltering_TypeDispatched<TDeviceType, scalar_t>(
                unprojected_points, unprojected_point_colors, depth_mask, raster_depth, raster_colors, intrinsics,
                extrinsics, depth_scale, depth_max, preserve_image_layout
        );
    } /*end lambda*/); // end macro call DISPATCH_DTYPE_TO_TEMPLATE
}

template<open3d::core::Device::DeviceType TDeviceType, typename TDepth>
void Unproject_TypeDispatched(
        open3d::core::Tensor& unprojected_points,
        const open3d::core::Tensor& projected_points, const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
        float depth_scale, float depth_max
) {
    o3c::Device device = projected_points.GetDevice();
    o3c::AssertTensorDtypes(projected_point_depth, { o3c::UInt16, o3c::Float32 });
    o3c::AssertTensorDevice(projected_points, device);
    o3c::AssertTensorShape(projected_points, { utility::nullopt, 2 });
    int64_t point_count = projected_points.GetShape(0);
    //TODO: upgrade to use cuda::std::function instead of boolean check once that is available. :Sadness: :..(
    // see https://nvidia.github.io/libcudacxx/standard_api/utility_library/functional.html
    bool use_uniform_depth = false;
    TDepth uniform_depth;
    if (projected_point_depth.NumElements() == 1) {
        o3c::AssertTensorShape(projected_point_depth, { 1 });
        uniform_depth = projected_point_depth.To(o3c::Device("CPU:0")).ToFlatVector<TDepth>()[0] / depth_scale;
        use_uniform_depth = true;
    } else {
        o3c::AssertTensorShape(projected_point_depth, { point_count });
        use_uniform_depth = false;
    }

    o3c::Tensor pose = open3d::t::geometry::InverseTransformation(extrinsics);
    o3tgk::TransformIndexer transform(intrinsics, pose, 1.0f);
    const auto projected_point_data = projected_points.GetDataPtr<float>();
    const auto projected_point_depth_data = projected_point_depth.GetDataPtr<float>();

    unprojected_points = o3c::Tensor({point_count, 3}, o3c::Float32, device);
    auto unprojected_point_data = unprojected_points.GetDataPtr<float>();

    NNRT_DECLARE_ATOMIC(int, unfiltered_point_count);

    o3c::ParallelFor(
            device, point_count,
            NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
                float depth_metric;
                if (use_uniform_depth) {
                    depth_metric = uniform_depth;
                } else {
                    depth_metric = projected_point_depth_data[workload_idx] / depth_scale;
                }
                if (depth_metric > 0 && depth_metric < depth_max) {
                    auto* projected_point = projected_point_data + workload_idx * 2;
                    int current_count = NNRT_ATOMIC_ADD(unfiltered_point_count, 1);
                    auto* unprojected_point = unprojected_point_data + current_count * 3;
                    float x_camera = 0, y_camera = 0, z_camera = 0;
                    transform.Unproject(projected_point[0], projected_point[1], depth_metric, &x_camera,
                                        &y_camera, &z_camera);
                    transform.RigidTransform(x_camera, y_camera, z_camera, unprojected_point + 0,
                                             unprojected_point + 1, unprojected_point + 2);

                }
            }
    );

    unprojected_points = unprojected_points.Slice(0, 0, NNRT_GET_ATOMIC_VALUE_HOST(unfiltered_point_count));
}

template<open3d::core::Device::DeviceType TDeviceType>
void Unproject(
        open3d::core::Tensor& unprojected_points,
        const open3d::core::Tensor& projected_points, const open3d::core::Tensor& projected_point_depth,
        const open3d::core::Tensor& intrinsics, const open3d::core::Tensor& extrinsics,
        float depth_scale, float depth_max
) {
    DISPATCH_DTYPE_TO_TEMPLATE(projected_point_depth.GetDtype(), [&]() {
        Unproject_TypeDispatched<TDeviceType, scalar_t>(
                unprojected_points, projected_points, projected_point_depth, intrinsics, extrinsics, depth_scale,
                depth_max
        );
    } /*end lambda*/); // end macro call DISPATCH_DTYPE_TO_TEMPLATE
}

} // namespace nnrt::geometry::functional::kernel