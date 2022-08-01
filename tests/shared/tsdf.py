from typing import Tuple
import open3d as o3d
import open3d.core as o3c
import nnrt
import numpy as np


def generate_xy_plane_depth_image(resolution: Tuple[int, int], depth: int) -> np.ndarray:
    image = np.ones(resolution, dtype=np.uint16) * depth
    return image


def generate_xy_plane_color_image(resolution: Tuple[int, int], value: Tuple[int, int, int]) -> np.ndarray:
    image = np.ndarray((resolution[0], resolution[1], 3), dtype=np.uint8)
    image[:, :] = value
    return image


def construct_intrinsic_matrix1_3x3():
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = 100.0
    intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = 50.0
    intrinsics[1, 2] = 50.0
    return intrinsics


def construct_intrinsic_matrix1_4x4():
    intrinsics = np.eye(4, dtype=np.float64)
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

    volume = nnrt.geometry.NonRigidSurfaceVoxelBlockGrid(
        ['tsdf', 'weight', 'color'],
        [o3d.core.Dtype.Float32, o3d.core.Dtype.UInt16, o3d.core.Dtype.UInt16],
        [o3d.core.SizeVector(1), o3d.core.SizeVector(1), o3d.core.SizeVector(1)],
        voxel_size=voxel_size,
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
    host = o3c.Device("CPU:0")
    intrinsics_open3d = o3c.Tensor(intrinsics, device=host)
    extrinsics_open3d = o3c.Tensor(np.eye(4, dtype=np.float64), device=host)

    # integrate volume
    blocks_to_activate = \
        volume.compute_unique_block_coordinates(depth_image_gpu, intrinsics_open3d, extrinsics_open3d, 1000.0, 3.0,
                                                trunc_voxel_multiplier=2.0)
    volume.integrate(blocks_to_activate, depth_image_gpu, color_image_gpu, intrinsics_open3d, extrinsics_open3d,
                     1000.0, 3.0, trunc_voxel_multiplier=2.0)
    return volume
