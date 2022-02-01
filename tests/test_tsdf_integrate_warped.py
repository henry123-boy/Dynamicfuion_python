import math
from typing import Tuple

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

import nnrt
import nnrt.geometry as nnrt_geom

from image_processing import compute_normals
from image_processing.numba_cuda.preprocessing import cuda_compute_normal
from image_processing.numpy_cpu.preprocessing import cpu_compute_normal


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


def construct_test_volume1(device=o3d.core.Device('cuda:0')):
    # initialize volume
    voxel_size = 0.01  # 1 cm voxel size
    sdf_truncation_distance = 0.02  # truncation distance = 2cm
    block_resolution = 8  # 8^3 voxel blocks
    initial_block_count = 128  # initially allocated number of voxel blocks

    volume = nnrt.geometry.WarpableTSDFVoxelGrid(
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


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_integrate_warped_simple_motion(device):
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

    # voxel size = 0.01 m
    volume = construct_test_volume1(device)
    voxel_tsdf_and_weights: o3c.Tensor = volume.extract_tsdf_values_and_weights()
    voxel_tsdf_and_weights_np_originals = voxel_tsdf_and_weights.cpu().numpy()

    # the first node moves 1 cm along the negative z axis (towards the camera).
    node_rotations_mat = np.array([np.eye(3, dtype=np.float32)] * (len(nodes)))
    node_translations_vec = np.array([[0.0, 0.0, -0.01]] + [[0.0, 0.0, 0.0]] * (len(nodes) - 1)).astype(np.float32)

    depth = 50  # mm
    image_width = 100
    image_height = 100
    image_resolution = (image_width, image_height)
    depth_image = generate_xy_plane_depth_image(image_resolution, depth)
    color_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

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
    normals = compute_normals(device, point_image)

    # ---- compute updates ----
    node_coverage = 0.05
    depth_image_o3d = o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(depth_image), device=device)
    color_image_o3d = o3d.t.geometry.Image.from_legacy(o3d.geometry.Image(color_image), device=device)
    normals_o3d = o3c.Tensor(normals, dtype=o3c.Dtype.Float32, device=device)
    intrinsic_matrix_o3d = o3c.Tensor(intrinsic_matrix, dtype=o3c.Dtype.Float32, device=device)
    extrinsic_matrix_o3d = o3c.Tensor.eye(4, dtype=o3c.Dtype.Float32, device=device)
    node_rotations_o3d = o3c.Tensor(node_rotations_mat, dtype=o3c.Dtype.Float32, device=device)
    node_translations_o3d = o3c.Tensor(node_translations_vec, dtype=o3c.Dtype.Float32, device=device)
    nodes_o3d = o3c.Tensor(nodes, dtype=o3c.Dtype.Float32, device=device)
    edges_o3d = o3c.Tensor([[1, 2, 3, 4],
                            [-1, -1, -1, -1],
                            [-1, -1, -1, -1],
                            [-1, -1, -1, -1],
                            [-1, -1, -1, -1]], device=device)
    edge_weights_o3d = o3c.Tensor.ones(edges_o3d.shape, dtype=o3c.Dtype.Float32, device=device)
    clusters = o3c.Tensor((0, 0, 0, 0, 0), device=device)

    # ---- construct warp-field ----
    warp_field = nnrt_geom.GraphWarpField(nodes_o3d, edges_o3d, edge_weights_o3d, clusters, node_coverage, True, 4, 3)
    warp_field.rotations = node_rotations_o3d
    warp_field.translations = node_translations_o3d

    cos_voxel_ray_to_normal = volume.integrate_warped(
        depth_image_o3d, color_image_o3d, normals_o3d, intrinsic_matrix_o3d, extrinsic_matrix_o3d, warp_field,
        depth_scale=1000.0, depth_max=3.0)

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
                       center_plane_voxel_index + 1,  # x + 1
                       center_plane_voxel_index + 8,  # y + 1
                       center_plane_voxel_index + 16,  # y + 2
                       center_plane_voxel_index + 64]  # z + 1

    # generated using the above function.
    # Note: if anything about the reference implementation changes, these residuals need to be re-computed.
    # each array row contains:
    # v, u, cosine, tsdf, weight
    ground_truth_data = np.array([
        [50, 50, 0.4970065653324127, 0.0, 0.0],
        [71, 50, 0.9784621335214618, 0.06499883711342021, 2.0],
        [50, 71, 0.9784621335214618, 0.06499883711342021, 2.0],
        [50, 92, 0.9215041958391356, 0.06362117264804237, 2.0],
        [50, 50, 0.4970065653324127, 0.0, 0.0]
    ])

    def check_voxel_at(index, ground_truth):
        assert math.isclose(cos_voxel_ray_to_normal[int(ground_truth[0]), int(ground_truth[1])], ground_truth[2],
                            abs_tol=1e-7)
        if ground_truth[2] > 0.5:
            assert np.allclose(voxel_tsdf_and_weights_np[index], ground_truth[3:])

    for index, ground_truth in zip(indices_to_test, ground_truth_data):
        check_voxel_at(index, ground_truth)
