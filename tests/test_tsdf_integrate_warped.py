import math

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
from dq3d import quat, dualquat

import os


import nnrt

from pipeline.numba_cuda.preprocessing import cuda_compute_normal
from pipeline.numpy_cpu.preprocessing import cpu_compute_normal
from tests.shared.tsdf import generate_xy_plane_depth_image, construct_test_volume1, construct_intrinsic_matrix1_3x3


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_integrate_warped_simple_motion_dq(device):
    camera_rotation = np.ascontiguousarray(np.eye(3, dtype=np.float32))
    camera_translation = np.ascontiguousarray(np.zeros(3, dtype=np.float32))
    # we need at least four nodes this time, otherwise psdf computation will consider voxel invalid and produce "NaN".
    # Make it five.
    nodes = np.array([[0.0, 0.0, 0.05],
                      [0.02, 0.0, 0.05],
                      [-0.02, 0.0, 0.05],
                      [0.00, 0.02, 0.05],
                      [0.00, -0.02, 0.05]],
                     dtype=np.float32)

    volume = construct_test_volume1(device)
    voxel_tsdf_and_weights: o3c.Tensor = volume.extract_tsdf_values_and_weights()
    voxel_tsdf_and_weights_np_originals = voxel_tsdf_and_weights.cpu().numpy()

    # the first node moves 1 cm along the negative z axis (towards the camera).
    node_dual_quaternions_dq3d = [dualquat(quat.identity(), quat(1.0, 0.0, 0.0, -0.005))] + [dualquat(quat.identity())] * (len(nodes) - 1)
    node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in node_dual_quaternions_dq3d])

    depth = 50  # mm
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)

    # let's imagine that the central surface point is 1 cm closer to the camera as well, so we alter the depth
    # to 40 mm there. Make the motion cease at the other four nodes, e.g. their depth should remain at 50.
    # We can make a radial "pinch" in the center of the depth image.
    # For our predefined camera, 1 px = 0.005 m, and the nodes are around the 0.002 m radius,
    # which puts our pixel radius at 0.002 / 0.0005 = 40 px
    pinch_diameter = 40
    pinch_radius = pinch_diameter // 2
    pinch_height = 10
    y_coordinates = np.linspace(-1, 1, pinch_diameter)[None, :] * pinch_height
    x_coordinates = np.linspace(-1, 1, pinch_diameter)[:, None] * pinch_height
    delta = -pinch_height + np.sqrt(x_coordinates ** 2 + y_coordinates ** 2)
    half_image_width = image_width // 2
    half_image_height = image_height // 2
    # @formatter:off
    depth_image[half_image_height - pinch_radius:half_image_height + pinch_radius,
    half_image_width - pinch_radius:half_image_width + pinch_radius] += np.round(delta).astype(np.uint16)
    # @formatter:on

    # ---- compute normals ----
    intrinsic_matrix = construct_intrinsic_matrix1_3x3()
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)
    normals = cuda_compute_normal(point_image)

    # ---- compute updates ----
    truncation_distance = 0.02  # same value as in construct_test_volume1
    node_coverage = 0.05
    depth_image_o3d = o3d.t.geometry.Image.from_legacy_image(o3d.geometry.Image(depth_image), device=device)
    normals_o3d = o3c.Tensor(normals, dtype=o3c.Dtype.Float32, device=device)
    intrinsic_matrix_o3d = o3c.Tensor(intrinsic_matrix, dtype=o3c.Dtype.Float32, device=device)
    extrinsic_matrix_o3d = o3c.Tensor.eye(4, dtype=o3c.Dtype.Float32, device=device)
    node_dual_quaternions_o3d = o3c.Tensor(node_dual_quaternions, dtype=o3c.Dtype.Float32, device=device)
    nodes_o3d = o3c.Tensor(nodes, dtype=o3c.Dtype.Float32)

    cos_voxel_ray_to_normal = volume.integrate_warped_dq(
        depth_image_o3d, normals_o3d, intrinsic_matrix_o3d, extrinsic_matrix_o3d,
        nodes_o3d, node_dual_quaternions_o3d, node_coverage,
        anchor_count=4, depth_scale=1000.0, depth_max=3.0)

    cos_voxel_ray_to_normal = np.squeeze(cos_voxel_ray_to_normal.cpu().numpy())

    voxel_tsdf_and_weights: o3c.Tensor = volume.extract_tsdf_values_and_weights()
    voxel_tsdf_and_weights_np = voxel_tsdf_and_weights.cpu().numpy()

    # voxel in the center of the plane is at 0, 0, 0.05,
    # which should coincide with the first and only node
    # voxel global position is (0, 0, 5) (in voxels)
    # voxel is, presumably, in block 3
    # voxel's index in block 0 is 5 * (8*8) = 320
    # each block holds 512 voxels

    center_plane_voxel_index = 512 + 512 + 512 + 320

    indices_to_test = [center_plane_voxel_index,
                       center_plane_voxel_index + 1,
                       center_plane_voxel_index + 8,
                       center_plane_voxel_index + 16,
                       center_plane_voxel_index + 64]

    # generated using the above function.
    # Note: if anything about the reference implementation changes, these values need to be re-computed.
    # each array row contains:
    # u, v, cosine, tsdf, weight
    ground_truth_data = np.array([
        [50, 50, 0.4970065653324127, 0.0, 0.0],
        [71, 50, 0.9784621335214618, 0.06499883711342021, 2.0],
        [50, 71, 0.9784621335214618, 0.06499883711342021, 2.0],
        [50, 92, 0.9215041958391356, 0.06362117264804237, 2.0],
        [50, 50, 0.4970065653324127, 0.0, 0.0]
    ])

    def check_voxel_at(index, ground_truth):
        assert math.isclose(cos_voxel_ray_to_normal[int(ground_truth[0]), int(ground_truth[1])], ground_truth[2], abs_tol=1e-7)
        if ground_truth[2] > 0.5:
            assert np.allclose(voxel_tsdf_and_weights_np[index], ground_truth[3:])

    for index, ground_truth in zip(indices_to_test, ground_truth_data):
        check_voxel_at(index, ground_truth)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_integrate_warped_simple_motion_mat(device):
    camera_rotation = np.ascontiguousarray(np.eye(3, dtype=np.float32))
    camera_translation = np.ascontiguousarray(np.zeros(3, dtype=np.float32))
    # we need at least four nodes this time, otherwise psdf computation will consider voxel invalid and produce "NaN".
    # Make it five.
    nodes = np.array([[0.0, 0.0, 0.05],
                      [0.02, 0.0, 0.05],
                      [-0.02, 0.0, 0.05],
                      [0.00, 0.02, 0.05],
                      [0.00, -0.02, 0.05]],
                     dtype=np.float32)

    volume = construct_test_volume1(device)
    voxel_tsdf_and_weights: o3c.Tensor = volume.extract_tsdf_values_and_weights()
    voxel_tsdf_and_weights_np_originals = voxel_tsdf_and_weights.cpu().numpy()

    # the first node moves 1 cm along the negative z axis (towards the camera).
    node_dual_quaternions_dq3d = [dualquat(quat.identity(), quat(1.0, 0.0, 0.0, -0.005))] + [dualquat(quat.identity())] * (len(nodes) - 1)
    node_rotations_mat = np.array([dq.rotation().to_rotation_matrix().astype(np.float32) for dq in node_dual_quaternions_dq3d])
    node_translations_vec = np.array([dq.translation().astype(np.float32) for dq in node_dual_quaternions_dq3d])

    depth = 50  # mm
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)

    # let's imagine that the central surface point is 1 cm closer to the camera as well, so we alter the depth
    # to 40 mm there. Make the motion cease at the other four nodes, e.g. their depth should remain at 50.
    # We can make a radial "pinch" in the center of the depth image.
    # For our predefined camera, 1 px = 0.005 m, and the nodes are around the 0.002 m radius,
    # which puts our pixel radius at 0.002 / 0.0005 = 40 px
    pinch_diameter = 40
    pinch_radius = pinch_diameter // 2
    pinch_height = 10
    y_coordinates = np.linspace(-1, 1, pinch_diameter)[None, :] * pinch_height
    x_coordinates = np.linspace(-1, 1, pinch_diameter)[:, None] * pinch_height
    delta = -pinch_height + np.sqrt(x_coordinates ** 2 + y_coordinates ** 2)
    half_image_width = image_width // 2
    half_image_height = image_height // 2
    # @formatter:off
    depth_image[half_image_height - pinch_radius:half_image_height + pinch_radius,
                half_image_width - pinch_radius:half_image_width + pinch_radius] += np.round(delta).astype(np.uint16)
    # @formatter:on

    # ---- compute normals ----
    intrinsic_matrix = construct_intrinsic_matrix1_3x3()
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)
    normals = cpu_compute_normal(point_image)

    # ---- compute updates ----
    truncation_distance = 0.02  # same value as in construct_test_volume1
    node_coverage = 0.05
    depth_image_o3d = o3d.t.geometry.Image.from_legacy_image(o3d.geometry.Image(depth_image), device=device)
    normals_o3d = o3c.Tensor(normals, dtype=o3c.Dtype.Float32, device=device)
    intrinsic_matrix_o3d = o3c.Tensor(intrinsic_matrix, dtype=o3c.Dtype.Float32, device=device)
    extrinsic_matrix_o3d = o3c.Tensor.eye(4, dtype=o3c.Dtype.Float32, device=device)
    node_rotations_o3d = o3c.Tensor(node_rotations_mat, dtype=o3c.Dtype.Float32, device=device)
    node_translations_o3d = o3c.Tensor(node_translations_vec, dtype=o3c.Dtype.Float32, device=device)
    nodes_o3d = o3c.Tensor(nodes, dtype=o3c.Dtype.Float32)

    cos_voxel_ray_to_normal = volume.integrate_warped_mat(
        depth_image_o3d, normals_o3d, intrinsic_matrix_o3d, extrinsic_matrix_o3d,
        nodes_o3d, node_rotations_o3d, node_translations_o3d, node_coverage,
        anchor_count=4, depth_scale=1000.0, depth_max=3.0)



    # __DEBUG (restore)
    #
    # cos_voxel_ray_to_normal = np.squeeze(cos_voxel_ray_to_normal.cpu().numpy())
    #
    # voxel_tsdf_and_weights: o3c.Tensor = volume.extract_tsdf_values_and_weights()
    # voxel_tsdf_and_weights_np = voxel_tsdf_and_weights.cpu().numpy()
    #
    # # voxel in the center of the plane is at 0, 0, 0.05,
    # # which should coincide with the first and only node
    # # voxel global position is (0, 0, 5) (in voxels)
    # # voxel is, presumably, in block 3
    # # voxel's index in block 0 is 5 * (8*8) = 320
    # # each block holds 512 voxels
    #
    # center_plane_voxel_index = 512 + 512 + 512 + 320
    #
    # indices_to_test = [center_plane_voxel_index,
    #                    center_plane_voxel_index + 1,
    #                    center_plane_voxel_index + 8,
    #                    center_plane_voxel_index + 16,
    #                    center_plane_voxel_index + 64]
    #
    # # generated using the above function.
    # # Note: if anything about the reference implementation changes, these values need to be re-computed.
    # # each array row contains:
    # # u, v, cosine, tsdf, weight
    # ground_truth_data = np.array([
    #     [50, 50, 0.4970065653324127, 0.0, 0.0],
    #     [71, 50, 0.9784621335214618, 0.06499883711342021, 2.0],
    #     [50, 71, 0.9784621335214618, 0.06499883711342021, 2.0],
    #     [50, 92, 0.9215041958391356, 0.06362117264804237, 2.0],
    #     [50, 50, 0.4970065653324127, 0.0, 0.0]
    # ])
    #
    # def check_voxel_at(index, ground_truth):
    #     assert math.isclose(cos_voxel_ray_to_normal[int(ground_truth[0]), int(ground_truth[1])], ground_truth[2], abs_tol=1e-7)
    #     if ground_truth[2] > 0.5:
    #         assert np.allclose(voxel_tsdf_and_weights_np[index], ground_truth[3:])
    #
    # for index, ground_truth in zip(indices_to_test, ground_truth_data):
    #     check_voxel_at(index, ground_truth)
