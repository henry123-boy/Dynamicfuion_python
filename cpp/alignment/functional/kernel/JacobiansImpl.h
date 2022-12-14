//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 11/21/22.
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
#include <open3d/t/geometry/Utility.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>


// local includes
#include "alignment/functional/kernel/Jacobians.h"
#include "alignment/functional/kernel/BarycentricCoordinateJacobians.h"
#include "alignment/functional/kernel/ProjectionJacobians.h"
#include "core/PlatformIndependentQualifiers.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace utility = open3d::utility;

namespace nnrt::alignment::functional::kernel {

template<open3d::core::Device::DeviceType TDeviceType>
void WarpedVertexAndNormalJacobians(open3d::core::Tensor& vertex_rotation_jacobians, open3d::core::Tensor& normal_rotation_jacobians,
                                    const open3d::core::Tensor& vertex_positions, const open3d::core::Tensor& vertex_normals,
                                    const open3d::core::Tensor& node_positions, const open3d::core::Tensor& node_rotations,
                                    const open3d::core::Tensor& warp_anchors, const open3d::core::Tensor& warp_anchor_weights) {
	auto device = vertex_positions.GetDevice();
	o3c::AssertTensorDevice(vertex_normals, device);
	o3c::AssertTensorDevice(node_positions, device);
	o3c::AssertTensorDevice(node_rotations, device);
	o3c::AssertTensorDevice(warp_anchors, device);
	o3c::AssertTensorDevice(warp_anchor_weights, device);
	o3c::AssertTensorShape(vertex_positions, { utility::nullopt, 3 });
	o3c::AssertTensorShape(vertex_normals, vertex_positions.GetShape());
	auto node_count = node_positions.GetLength();
	o3c::AssertTensorShape(node_rotations, { node_count, 3, 3 });
	auto vertex_count = vertex_positions.GetLength();
	o3c::AssertTensorShape(warp_anchors, { vertex_count, utility::nullopt });
	o3c::AssertTensorShape(warp_anchor_weights, warp_anchors.GetShape());
	auto anchors_per_vertex = warp_anchors.GetShape(1);
	o3c::AssertTensorDtype(warp_anchors, o3c::Int32);
	o3c::AssertTensorDtype(warp_anchor_weights, o3c::Float32);
	o3c::AssertTensorDtype(vertex_positions, o3c::Float32);
	o3c::AssertTensorDtype(vertex_normals, o3c::Float32);
	o3c::AssertTensorDtype(node_positions, o3c::Float32);
	o3c::AssertTensorDtype(node_rotations, o3c::Float32);


	const auto* warp_anchor_data = warp_anchors.template GetDataPtr<int32_t>();
	const auto* warp_anchor_weight_data = warp_anchor_weights.template GetDataPtr<float>();

	const auto* vertex_position_data = vertex_positions.GetDataPtr<float>();
	const auto* vertex_normal_data = vertex_normals.GetDataPtr<float>();
	const auto* node_position_data = node_positions.GetDataPtr<float>();
	const auto* node_rotation_data = node_rotations.GetDataPtr<float>();

	// these will be used as skew-symmetric vectors later
	vertex_rotation_jacobians = o3c::Tensor({vertex_count, anchors_per_vertex, 4}, vertex_positions.GetDtype(), device);
	normal_rotation_jacobians = o3c::Tensor({vertex_count, anchors_per_vertex, 3}, vertex_positions.GetDtype(), device);

	auto* vertex_jacobian_data = vertex_rotation_jacobians.GetDataPtr<float>();
	auto* normal_jacobian_data = normal_rotation_jacobians.GetDataPtr<float>();

	o3c::ParallelFor(
			device, vertex_count * anchors_per_vertex,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				const auto i_vertex = workload_idx / anchors_per_vertex;
				const auto i_anchor = workload_idx % anchors_per_vertex;
				const auto i_node = warp_anchor_data[i_vertex * anchors_per_vertex + i_anchor];
				const auto node_weight = warp_anchor_weight_data[i_vertex * anchors_per_vertex + i_anchor];
				Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> node_rotation(node_rotation_data + (i_node * 9));
				Eigen::Map<const Eigen::Vector3f> node_position(node_position_data + (i_node * 3));
				Eigen::Map<const Eigen::Vector3f> vertex_position(vertex_position_data + (i_vertex * 3));
				Eigen::Map<const Eigen::Vector3f> vertex_normal(vertex_normal_data + (i_vertex * 3));

				Eigen::Map<Eigen::Vector3f> vertex_rotation_jacobian(vertex_jacobian_data + (i_vertex * anchors_per_vertex * 4) + (i_anchor * 4));
				// store node weight to increase speed of retrieval for the vertex translation jacobian
				float* stored_node_weight = vertex_jacobian_data + (i_vertex * anchors_per_vertex * 4) + (i_anchor * 4) + 3;
				*stored_node_weight = -node_weight;
				Eigen::Map<Eigen::Vector3f> normal_rotation_jacobian(normal_jacobian_data + (i_vertex * anchors_per_vertex * 3) + (i_anchor * 3));

				vertex_rotation_jacobian =
						-node_weight * (node_rotation * (vertex_position - node_position));
				normal_rotation_jacobian = -node_weight * (node_rotation * vertex_normal);
			}
	);

}

template<open3d::core::Device::DeviceType TDeviceType, bool TWithPerspectiveCorrection,
		rendering::functional::kernel::FrontFaceVertexOrder TVertexOrder = rendering::functional::kernel::ClockWise>
void RenderedVertexAndNormalJacobians(open3d::core::Tensor& rendered_vertex_jacobians, open3d::core::Tensor& rendered_normal_jacobians,
                                      const open3d::core::Tensor& warped_vertex_positions, const open3d::core::Tensor& warped_triangle_indices,
                                      const open3d::core::Tensor& warped_vertex_normals, const open3d::core::Tensor& pixel_faces,
                                      const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsics) {
	auto device = warped_vertex_positions.GetDevice();
	o3c::AssertTensorDevice(warped_triangle_indices, device);
	o3c::AssertTensorDevice(warped_vertex_normals, device);
	o3c::AssertTensorDevice(pixel_faces, device);
	o3c::AssertTensorDevice(pixel_barycentric_coordinates, device);

	o3tg::CheckIntrinsicTensor(ndc_intrinsics);
	o3tgk::TransformIndexer perspective_projection(ndc_intrinsics, o3c::Tensor::Eye(4, o3c::Float64, o3c::Device("CPU:0")), 1.0f);

	const auto image_height = pixel_faces.GetShape(0);
	const auto image_width = pixel_faces.GetShape(1);
	const auto image_height_int = static_cast<int32_t>(image_height);
	const auto image_width_int = static_cast<int32_t>(image_width);
	const auto faces_per_pixel = pixel_faces.GetShape(2);
	const auto vertex_count = warped_vertex_positions.GetLength();

	o3c::AssertTensorShape(warped_vertex_positions, { vertex_count, 3 });
	o3c::AssertTensorShape(warped_vertex_normals, { vertex_count, 3 });
	o3c::AssertTensorShape(warped_triangle_indices, { utility::nullopt, 3 });
	o3c::AssertTensorShape(pixel_barycentric_coordinates, { image_height, image_width, faces_per_pixel, 3 });
	o3c::AssertTensorShape(pixel_faces, { image_height, image_width, faces_per_pixel });

	o3c::AssertTensorDtype(warped_vertex_positions, o3c::Float32);
	o3c::AssertTensorDtype(warped_vertex_normals, o3c::Float32);
	o3c::AssertTensorDtype(pixel_barycentric_coordinates, o3c::Float32);
	o3c::AssertTensorDtype(pixel_faces, o3c::Int64);

	auto pixel_count = image_height * image_width;

	auto triangle_index_data = warped_triangle_indices.template GetDataPtr<int64_t>();
	auto vertex_position_data = warped_vertex_positions.template GetDataPtr<float>();
	auto vertex_normal_data = warped_vertex_normals.template GetDataPtr<float>();

	auto pixel_face_data = pixel_faces.template GetDataPtr<int64_t>();
	auto barycentric_coordinate_data = pixel_barycentric_coordinates.template GetDataPtr<float>();


	rendered_vertex_jacobians = o3c::Tensor({image_height, image_width, 3, 9}, o3c::Float32, device);
	rendered_normal_jacobians = o3c::Tensor({image_height, image_width, 3, 10}, o3c::Float32, device);
	auto rendered_vertex_jacobian_data = rendered_vertex_jacobians.GetDataPtr<float>();
	auto rendered_normal_jacobian_data = rendered_normal_jacobians.GetDataPtr<float>();


	o3c::ParallelFor(
			device, pixel_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				auto v_image = static_cast<int>(workload_idx / image_width);
				auto u_image =  static_cast<int>(workload_idx % image_width);
				const float y_screen = rendering::kernel::ImageSpacePixelAlongDimensionToNdc(v_image, image_height_int,
                                                                                             image_width_int);
				const float x_screen = rendering::kernel::ImageSpacePixelAlongDimensionToNdc(u_image, image_width_int,
                                                                                             image_height_int);
				Eigen::Vector2f ray_point(x_screen, y_screen);

				auto i_face = pixel_face_data[(v_image * image_width * faces_per_pixel) + (u_image * faces_per_pixel)];

				if (i_face == -1) {
					return;
				}

				Eigen::Map<const Eigen::Matrix<int64_t, 3, 1>> face_vertex_indices(triangle_index_data + i_face * 3);

				Eigen::Map<const Eigen::Vector3f> face_vertex0(vertex_position_data + face_vertex_indices(0) * 3);
				Eigen::Map<const Eigen::Vector3f> face_vertex1(vertex_position_data + face_vertex_indices(1) * 3);
				Eigen::Map<const Eigen::Vector3f> face_vertex2(vertex_position_data + face_vertex_indices(2) * 3);
				Matrix3f face_vertex_matrix;
				face_vertex_matrix << face_vertex0, face_vertex1, face_vertex2;


				Eigen::Map<const Eigen::Vector3f> face_normal0(vertex_normal_data + face_vertex_indices(0) * 3);
				Eigen::Map<const Eigen::Vector3f> face_normal1(vertex_normal_data + face_vertex_indices(1) * 3);
				Eigen::Map<const Eigen::Vector3f> face_normal2(vertex_normal_data + face_vertex_indices(2) * 3);
				Matrix3f face_normal_matrix;
				face_normal_matrix << face_normal0, face_normal1, face_normal2;


				Matrix3x9f barycentric_coordinate_jacobian;

				auto barycentric_coordinates_index = (v_image * image_width * faces_per_pixel * 3) + (u_image * faces_per_pixel * 3);
				Eigen::Map<const Eigen::RowVector3f> barycentric_coordinates(barycentric_coordinate_data + barycentric_coordinates_index);

				if (TWithPerspectiveCorrection) {
					barycentric_coordinate_jacobian =
							Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_WithPerspectiveCorrection<TVertexOrder>(
									ray_point, face_vertex0, face_vertex1, face_vertex2, perspective_projection
							);
				} else {
					// avoid recomputing distorted barycentric coordinates
					barycentric_coordinate_jacobian =
							Jacobian_BarycentricCoordinatesWrtCameraSpaceVertices_WithoutPerspectiveCorrection<TVertexOrder>(
									ray_point, face_vertex0, face_vertex1, face_vertex2, perspective_projection, barycentric_coordinates
							);
				}

				Eigen::Map<Matrix3x9f> pixel_rendered_vertex_jacobian_wrt_face_vertices(rendered_vertex_jacobian_data + workload_idx * (3*9));
				Eigen::Map<Matrix3x9f> pixel_rendered_normal_jacobian_wrt_face_vertices(rendered_normal_jacobian_data + workload_idx * (3*10));
				Eigen::Map<Eigen::RowVector3f> barycentric_coordinates_out(rendered_normal_jacobian_data + workload_idx * (3*10) + (3*9));

				pixel_rendered_vertex_jacobian_wrt_face_vertices =
						face_vertex_matrix * barycentric_coordinate_jacobian + Eigen::kroneckerProduct(barycentric_coordinates, Matrix3f::Identity());

				pixel_rendered_normal_jacobian_wrt_face_vertices =
						face_normal_matrix * barycentric_coordinate_jacobian;
				// this will be used to compute ∂(ρn)/∂n later, which is just ρ ⊗ 𝕀_3x3
				barycentric_coordinates_out = barycentric_coordinates;
			}
	);
}

template<open3d::core::Device::DeviceType TDeviceType>
void RenderedVertexAndNormalJacobians(open3d::core::Tensor& rendered_vertex_jacobians, open3d::core::Tensor& rendered_normal_jacobians,
                                      const open3d::core::Tensor& warped_vertex_positions, const open3d::core::Tensor& warped_triangle_indices,
                                      const open3d::core::Tensor& warped_vertex_normals, const open3d::core::Tensor& pixel_faces,
                                      const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& ndc_intrinsics,
                                      bool perspective_corrected_barycentric_coordinates) {
	if (perspective_corrected_barycentric_coordinates) {
		RenderedVertexAndNormalJacobians<TDeviceType, true>(
				rendered_normal_jacobians, rendered_normal_jacobians, warped_vertex_positions, warped_triangle_indices,
				warped_vertex_normals, pixel_faces, pixel_barycentric_coordinates, ndc_intrinsics
		);
	} else {
		RenderedVertexAndNormalJacobians<TDeviceType, false>(
				rendered_normal_jacobians, rendered_normal_jacobians, warped_vertex_positions, warped_triangle_indices,
				warped_vertex_normals, pixel_faces, pixel_barycentric_coordinates, ndc_intrinsics
		);
	}
}

} // namespace nnrt::alignment::functional::kernel

