import math

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import nnrt
from dq3d import quat, dualquat, op

from typing import Tuple

from pipeline.numba_cuda.fusion_functions import cuda_compute_psdf_voxel_centers, cuda_compute_voxel_center_anchors


def generate_xy_plane_depth_image(resolution: Tuple[int, int], depth: int) -> np.ndarray:
    image = np.ones(resolution, dtype=np.uint16) * depth
    return image


def generate_xy_plane_color_image(resolution: Tuple[int, int], value: Tuple[int, int, int]) -> np.ndarray:
    image = np.ndarray((resolution[0], resolution[1], 3), dtype=np.uint8)
    image[:, :] = value
    return image


def construct_intrinsic_matrix1_3x3():
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = 100.0
    intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = 50.0
    intrinsics[1, 2] = 50.0
    return intrinsics


def construct_intrinsic_matrix1_4x4():
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 0] = 100.0
    intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = 50.0
    intrinsics[1, 2] = 50.0
    return intrinsics


def construct_test_volume1(device=o3d.core.Device('cpu:0')):
    # initialize volume
    voxel_size = 0.01  # 1 cm voxel size
    sdf_truncation_distance = 0.02  # truncation distance = 2cm
    block_resolution = 8  # 8^3 voxel blocks
    initial_block_count = 128  # initially allocated number of voxel blocks

    volume = nnrt.geometry.ExtendedTSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_truncation_distance,
        block_resolution=block_resolution,
        block_count=initial_block_count,
        device=device)

    # generate image
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth = 50  # mm
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)
    depth_image_gpu = o3d.t.geometry.Image(o3c.Tensor(depth_image, device=device))
    value = (100, 100, 100)
    color_image = generate_xy_plane_color_image(image_resolution, value)
    color_image_gpu = o3d.t.geometry.Image(o3c.Tensor(color_image, device=device))

    # set up matrix parameters
    intrinsics = construct_intrinsic_matrix1_3x3()
    intrinsics_open3d_gpu = o3c.Tensor(intrinsics, device=device)
    extrinsics_open3d_gpu = o3c.Tensor(np.eye(4, dtype=np.float32), device=device)

    # integrate volume
    volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_open3d_gpu, extrinsics_open3d_gpu, 1000.0, 3.0)
    return volume


def test_tsdf_value_extraction():
    volume = construct_test_volume1()
    values = volume.extract_values_in_extent(-2, -2, 3, 3, 3, 8)

    values_np = values.cpu().numpy()

    expected_first_slice = np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]], dtype=np.float32)

    # we take out the [1:4, 1:4] portion of the first & second slices because the border values
    # will not be set (they are expected to be missed by rays projected from the camera)
    assert np.allclose(values_np[0, 1:4, 1:4], expected_first_slice, atol=1e-6)

    expected_second_slice = np.array([[0.5, 0.5, 0.5],
                                      [0.5, 0.5, 0.5],
                                      [0.5, 0.5, 0.5]], dtype=np.float32)

    assert np.allclose(values_np[1, 1:4, 1:4], expected_second_slice, atol=1e-6)

    expected_third_slice = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    assert np.allclose(values_np[2, :, :], expected_third_slice, atol=1e-6)

    expected_fourth_slice = np.array([[-0.5, -0.5, -0.5, -0.5, -0.5],
                                      [-0.5, -0.5, -0.5, -0.5, -0.5],
                                      [-0.5, -0.5, -0.5, -0.5, -0.5],
                                      [-0.5, -0.5, -0.5, -0.5, -0.5],
                                      [-0.5, -0.5, -0.5, -0.5, -0.5]], dtype=np.float32)

    assert np.allclose(values_np[3, :, :], expected_fourth_slice, atol=1e-6)

    expected_fifth_slice = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0],
                                     [-1.0, -1.0, -1.0, -1.0, -1.0],
                                     [-1.0, -1.0, -1.0, -1.0, -1.0],
                                     [-1.0, -1.0, -1.0, -1.0, -1.0],
                                     [-1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float32)

    assert np.allclose(values_np[4, :, :], expected_fifth_slice, atol=1e-6)


def test_voxel_center_extraction():
    volume = construct_test_volume1()
    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    assert voxel_centers_np.shape == (2048, 3)
    extent_min = voxel_centers_np.min(axis=0)
    extent_max = voxel_centers_np.max(axis=0)
    assert np.allclose(extent_min, [-0.08, -0.08, 0.0])
    assert np.allclose(extent_max, [0.07, 0.07, 0.07])


def test_compute_voxel_center_anchors():
    camera_rotation = np.ascontiguousarray(np.eye(3, dtype=np.float32))
    camera_translation = np.ascontiguousarray(np.zeros(3, dtype=np.float32))
    nodes = np.array([[0.0, 0.0, 0.05],
                      [0.02, 0.0, 0.05]],
                     dtype=np.float32)

    volume = construct_test_volume1()
    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    node_coverage = 0.05
    voxel_center_anchors, voxel_center_weights = \
        cuda_compute_voxel_center_anchors(voxel_centers_np, nodes, camera_rotation, camera_translation, node_coverage)

    # voxel in the center of the plane is at 0, 0, 0.05,
    # which should coincide with the first and only node
    # voxel index is (0, 0, 5)
    # voxel is, presumably, in block 3
    # voxel's index in block 0 is 5 * (8*8) = 320
    # each block holds 512 voxels

    center_plane_voxel_index = 512 + 512 + 512 + 320

    def get_weight(nc, distance):
        return math.exp(-distance ** 2 / (2 * nc * nc))

    w0 = get_weight(node_coverage, 0.0)
    w1 = get_weight(node_coverage, 0.02)  # 2-nd node is 2 cm away to the right
    expected_weights_center = np.array([[w0 / (w0 + w1), w1 / (w0 + w1), 0.0, 0.0]])
    expected_weights_x_plus_two = np.array([w1 / (w0 + w1), w0 / (w0 + w1), 0.0, 0.0])

    # the voxel at center_plane_voxel_index should coincide with the location of the first node
    assert np.allclose(voxel_centers_np[center_plane_voxel_index], nodes[0])
    assert np.allclose(voxel_center_anchors[center_plane_voxel_index], [0, 1, -1, -1])
    assert np.allclose(voxel_center_weights[center_plane_voxel_index], expected_weights_center)

    voxel_x_plus_two_index = center_plane_voxel_index + 2
    voxel_y_plus_two_index = center_plane_voxel_index + 16
    voxel_z_plus_two_index = center_plane_voxel_index + 128

    assert np.allclose(voxel_centers_np[voxel_x_plus_two_index], [0.02, 0.00, 0.05])
    assert np.allclose(voxel_centers_np[voxel_y_plus_two_index], [0.00, 0.02, 0.05])
    assert np.allclose(voxel_centers_np[voxel_z_plus_two_index], [0.00, 0.00, 0.07])

    assert np.allclose(voxel_center_anchors[voxel_x_plus_two_index], [0, 1, -1, -1])
    assert np.allclose(voxel_center_anchors[voxel_y_plus_two_index], [0, 1, -1, -1])
    assert np.allclose(voxel_center_anchors[voxel_z_plus_two_index], [0, 1, -1, -1])

    assert np.allclose(voxel_center_weights[voxel_x_plus_two_index], expected_weights_x_plus_two)
    dist0 = 0.02
    dist1 = np.linalg.norm(voxel_centers_np[voxel_y_plus_two_index] - nodes[1])
    w0 = get_weight(node_coverage, dist0)
    w1 = get_weight(node_coverage, dist1)

    expected_weights_remaining_pts = np.array([[w0 / (w0 + w1), w1 / (w0 + w1), 0.0, 0.0]])
    assert np.allclose(voxel_center_weights[voxel_y_plus_two_index], expected_weights_remaining_pts)
    assert np.allclose(voxel_center_weights[voxel_z_plus_two_index], expected_weights_remaining_pts)


def test_compute_psdf_voxel_centers_static():
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

    volume = construct_test_volume1()
    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    node_coverage = 0.05
    voxel_center_anchors, voxel_center_weights = \
        cuda_compute_voxel_center_anchors(voxel_centers_np, nodes, camera_rotation, camera_translation, node_coverage)

    # no motion at all for this case
    node_dual_quaternions = [dualquat(quat.identity())] * len(nodes)
    node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in node_dual_quaternions])

    depth = 50  # mm
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)

    intrinsic_matrix = construct_intrinsic_matrix1_4x4()

    voxel_psdf = cuda_compute_psdf_voxel_centers(depth_image, intrinsic_matrix, camera_rotation, camera_translation,
                                                 voxel_centers_np, voxel_center_anchors, voxel_center_weights,
                                                 node_dual_quaternions)
    # voxel in the center of the plane is at 0, 0, 0.05,
    # which should coincide with the first and only node
    # voxel index is (0, 0, 5)
    # voxel is, presumably, in block 3
    # voxel's index in block 0 is 5 * (8*8) = 320
    # each block holds 512 voxels

    center_plane_voxel_index = 512 + 512 + 512 + 320
    # the voxel at center_plane_voxel_index should coincide with the location of the first node
    assert np.allclose(voxel_centers_np[center_plane_voxel_index], nodes[0])
    assert math.isclose(voxel_psdf[center_plane_voxel_index], 0.0, abs_tol=1e-8)
    assert math.isclose(voxel_psdf[center_plane_voxel_index], 0.0, abs_tol=1e-8)

    voxel_x_plus_two_index = center_plane_voxel_index + 2
    voxel_y_plus_two_index = center_plane_voxel_index + 16  # as dictated by 8^3 block size
    voxel_z_plus_one_index = center_plane_voxel_index + 64

    assert np.allclose(voxel_centers_np[voxel_x_plus_two_index], [0.02, 0.00, 0.05])
    assert np.allclose(voxel_centers_np[voxel_y_plus_two_index], [0.00, 0.02, 0.05])
    assert np.allclose(voxel_centers_np[voxel_z_plus_one_index], [0.00, 0.00, 0.06])

    assert math.isclose(voxel_psdf[voxel_x_plus_two_index], 0.0, abs_tol=1e-8)
    assert math.isclose(voxel_psdf[voxel_y_plus_two_index], 0.0, abs_tol=1e-8)
    # 0.01 distance beyond the surface should yield a projective signed
    # distance of ~ -0.01 (note that it's not yet normalized to the truncation range)
    assert math.isclose(voxel_psdf[voxel_z_plus_one_index], -0.01, abs_tol=1e-8)


def test_compute_psdf_voxel_centers_static():
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

    volume = construct_test_volume1()
    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    node_coverage = 0.05
    voxel_center_anchors, voxel_center_weights = \
        cuda_compute_voxel_center_anchors(voxel_centers_np, nodes, camera_rotation, camera_translation, node_coverage)

    # no motion at all for this case
    node_dual_quaternions = [dualquat(quat.identity())] * len(nodes)
    node_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in node_dual_quaternions])

    depth = 50  # mm
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)

    intrinsic_matrix = construct_intrinsic_matrix1_4x4()

    voxel_psdf = cuda_compute_psdf_voxel_centers(depth_image, intrinsic_matrix, camera_rotation, camera_translation,
                                                 voxel_centers_np, voxel_center_anchors, voxel_center_weights,
                                                 node_dual_quaternions)
    # voxel in the center of the plane is at 0, 0, 0.05,
    # which should coincide with the first and only node
    # voxel index is (0, 0, 5)
    # voxel is, presumably, in block 3
    # voxel's index in block 0 is 5 * (8*8) = 320
    # each block holds 512 voxels

    center_plane_voxel_index = 512 + 512 + 512 + 320
    # the voxel at center_plane_voxel_index should coincide with the location of the first node
    assert np.allclose(voxel_centers_np[center_plane_voxel_index], nodes[0])
    assert math.isclose(voxel_psdf[center_plane_voxel_index], 0.0, abs_tol=1e-8)

    voxel_x_plus_two_index = center_plane_voxel_index + 2
    voxel_y_plus_two_index = center_plane_voxel_index + 16  # as dictated by 8^3 block size
    voxel_z_plus_one_index = center_plane_voxel_index + 64

    assert np.allclose(voxel_centers_np[voxel_x_plus_two_index], [0.02, 0.00, 0.05])
    assert np.allclose(voxel_centers_np[voxel_y_plus_two_index], [0.00, 0.02, 0.05])
    assert np.allclose(voxel_centers_np[voxel_z_plus_one_index], [0.00, 0.00, 0.06])

    assert math.isclose(voxel_psdf[voxel_x_plus_two_index], 0.0, abs_tol=1e-8)
    assert math.isclose(voxel_psdf[voxel_y_plus_two_index], 0.0, abs_tol=1e-8)
    # 0.01 distance beyond the surface should yield a projective signed
    # distance of ~ -0.01 (note that it's not yet normalized to the truncation range)
    assert math.isclose(voxel_psdf[voxel_z_plus_one_index], -0.01, abs_tol=1e-8)


def test_compute_psdf_voxel_centers_simple_motion():
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

    volume = construct_test_volume1()
    voxel_centers: o3c.Tensor = volume.extract_voxel_centers()
    voxel_centers_np = voxel_centers.cpu().numpy()

    node_coverage = 0.05
    voxel_center_anchors, voxel_center_weights = \
        cuda_compute_voxel_center_anchors(voxel_centers_np, nodes, camera_rotation, camera_translation, node_coverage)

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

    intrinsic_matrix = construct_intrinsic_matrix1_4x4()

    print()
    print(node_dual_quaternions[0])

    voxel_psdf = cuda_compute_psdf_voxel_centers(depth_image, intrinsic_matrix, camera_rotation, camera_translation,
                                                 voxel_centers_np, voxel_center_anchors, voxel_center_weights,
                                                 node_dual_quaternions)

    print(node_dual_quaternions[0])

    # voxel in the center of the plane is at 0, 0, 0.05,
    # which should coincide with the first and only node
    # voxel index is (0, 0, 5)
    # voxel is, presumably, in block 3
    # voxel's index in block 0 is 5 * (8*8) = 320
    # each block holds 512 voxels

    center_plane_voxel_index = 512 + 512 + 512 + 320
    # the voxel at center_plane_voxel_index should coincide with the location of the first node
    assert np.allclose(voxel_centers_np[center_plane_voxel_index], nodes[0])
    blended_dual_quaternion = op.dlb(voxel_center_weights[center_plane_voxel_index], node_dual_quaternions_dq3d)
    expected_deformed_z = blended_dual_quaternion.transform_point(nodes[0])[2]
    expected_depth = depth_image[50, 50] / 1000.
    expected_psdf = expected_depth - expected_deformed_z
    assert math.isclose(voxel_psdf[center_plane_voxel_index], expected_psdf, abs_tol=1e-8)

    voxel_x_plus_two_index = center_plane_voxel_index + 2
    voxel_y_plus_two_index = center_plane_voxel_index + 16  # as dictated by 8^3 block size
    voxel_z_plus_one_index = center_plane_voxel_index + 64

    indices = [voxel_x_plus_two_index, voxel_y_plus_two_index, voxel_z_plus_one_index]

    assert np.allclose(voxel_centers_np[voxel_x_plus_two_index], [0.02, 0.00, 0.05])
    assert np.allclose(voxel_centers_np[voxel_y_plus_two_index], [0.00, 0.02, 0.05])
    assert np.allclose(voxel_centers_np[voxel_z_plus_one_index], [0.00, 0.00, 0.06])

    blended_dual_quaternions = [op.dlb(voxel_center_weights[index], node_dual_quaternions_dq3d) for index in indices]
    points = [voxel_centers_np[index] for index in indices]

    deformed_points = [dq.transform_point(p) for dq, p in zip(blended_dual_quaternions, points)]
    psdfs = [voxel_psdf[index] for index in indices]
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    for i_point in range(len(indices)):
        deformed_point_x, deformed_point_y, deformed_point_z = deformed_points[i_point]
        du = int(round(fx * (deformed_point_x / deformed_point_z) + cx))
        dv = int(round(fy * (deformed_point_y / deformed_point_z) + cy))
        depth = depth_image[dv, du] / 1000.
        expected_psdf = depth - deformed_point_z
        psdf = psdfs[i_point]
        assert math.isclose(psdf, expected_psdf, abs_tol=1e-8)
