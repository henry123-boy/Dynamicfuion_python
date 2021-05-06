import numpy as np
import open3d as o3d
import open3d.core as o3c
import nnrt
from typing import Tuple


def generate_xy_plane_depth_image(resolution: Tuple[int, int], depth: int) -> np.ndarray:
    image = np.ones(resolution, dtype=np.uint16) * depth
    return image


def generate_xy_plane_color_image(resolution: Tuple[int, int], value: Tuple[int, int, int]) -> np.ndarray:
    image = np.ndarray((resolution[0], resolution[1], 3), dtype=np.uint8)
    image[:, :] = value
    return image


def construct_test_volume1():
    # device = o3d.core.Device('cuda:0')
    device = o3d.core.Device('cpu:0')

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
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = 100.0
    intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = 50.0
    intrinsics[1, 2] = 50.0
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