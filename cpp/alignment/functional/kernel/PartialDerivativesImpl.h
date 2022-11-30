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

// local includes
#include "rendering/functional/kernel/PartialDerivatives.h"
#include "core/PlatformIndependence.h"

namespace o3c = open3d::core;
namespace o3tg = open3d::t::geometry;
namespace utility = open3d::utility;

namespace nnrt::rendering::functional::kernel {

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

template<open3d::core::Device::DeviceType TDeviceType>
void RenderedVertexAndNormalJacobians(open3d::core::Tensor& rendered_vertex_jacobians, open3d::core::Tensor& rendered_normal_jacobians,
                                      const open3d::core::Tensor& warped_vertex_positions, const open3d::core::Tensor& warped_triangle_indices,
                                      const open3d::core::Tensor& warped_vertex_normals, const open3d::core::Tensor& pixel_faces,
                                      const open3d::core::Tensor& pixel_barycentric_coordinates, const open3d::core::Tensor& ray_space_intrinsics,
                                      bool perspective_corrected_barycentric_coordinates) {
	auto device = warped_vertex_positions.GetDevice();
	o3c::AssertTensorDevice(warped_triangle_indices, device);
	o3c::AssertTensorDevice(warped_vertex_normals, device);
	o3c::AssertTensorDevice(pixel_faces, device);
	o3c::AssertTensorDevice(pixel_barycentric_coordinates, device);

	o3tg::CheckIntrinsicTensor(ray_space_intrinsics);
	auto device_ray_space_intrinsics = ray_space_intrinsics.To(device).To(o3c::Float32);
	auto image_height = pixel_faces.GetShape(0);
	auto image_width = pixel_faces.GetShape(1);
	auto faces_per_pixel = pixel_faces.GetShape(2);
	auto vertex_count = warped_vertex_positions.GetLength();

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
	rendered_normal_jacobians = o3c::Tensor({image_height, image_width, 3, 9}, o3c::Float32, device);


	o3c::ParallelFor(
			device, pixel_count,
			NNRT_LAMBDA_CAPTURE_CLAUSE NNRT_DEVICE_WHEN_CUDACC(int64_t workload_idx) {
				int64_t v = workload_idx / image_width;
				int64_t u = workload_idx % image_width;
				auto i_face = pixel_face_data[(v * image_width * faces_per_pixel) + (u * faces_per_pixel)];

				if (i_face == -1) {
					return;
				}

				Eigen::Map<const Eigen::Matrix<int64_t, 3, 1>> face_vertex_indices(triangle_index_data + i_face * 3);

				Eigen::Map<const Eigen::Vector3f> face_vertex0(vertex_position_data + face_vertex_indices(0) * 3);
				Eigen::Map<const Eigen::Vector3f> face_vertex1(vertex_position_data + face_vertex_indices(1) * 3);
				Eigen::Map<const Eigen::Vector3f> face_vertex2(vertex_position_data + face_vertex_indices(2) * 3);
				Eigen::Map<const Eigen::Vector3f> face_vertices[] = {face_vertex0, face_vertex1, face_vertex2};

				Eigen::Map<const Eigen::Vector3f> face_normal0(vertex_normal_data + face_vertex_indices(0) * 3);
				Eigen::Map<const Eigen::Vector3f> face_normal1(vertex_normal_data + face_vertex_indices(1) * 3);
				Eigen::Map<const Eigen::Vector3f> face_normal2(vertex_normal_data + face_vertex_indices(2) * 3);
				Eigen::Map<const Eigen::Vector3f> face_normals[] = {face_normal0, face_normal1, face_normal2};

				auto barycentric_coordinates_index = (v * image_width * faces_per_pixel * 3) + (u * faces_per_pixel * 3);
				Eigen::Map<const Eigen::Vector3f> barycentric_coordinate0(barycentric_coordinate_data + barycentric_coordinates_index + 0 * 3);
				Eigen::Map<const Eigen::Vector3f> barycentric_coordinate1(barycentric_coordinate_data + barycentric_coordinates_index + 1 * 3);
				Eigen::Map<const Eigen::Vector3f> barycentric_coordinate2(barycentric_coordinate_data + barycentric_coordinates_index + 2 * 3);
				Eigen::Map<const Eigen::Vector3f> pixel_barycentric_coordinates[] =
						{barycentric_coordinate0, barycentric_coordinate1, barycentric_coordinate2};

				for (int i_vertex = 0; i_vertex < 3; i_vertex++) {
					Eigen::Map<const Eigen::Vector3f> face_vertex = face_vertices[i_vertex];
					Eigen::Map<const Eigen::Vector3f> face_normal = face_normals[i_vertex];
					Eigen::Map<const Eigen::Vector3f> pixel_barycentric_coordinate = pixel_barycentric_coordinates[i_vertex];




				}


			}
	);

}

} // namespace nnrt::rendering::functional::kernel

