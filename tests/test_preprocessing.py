import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

import nnrt
import data.camera
from data import StandaloneFrameDataset, StandaloneFramePreset
from image_processing import compute_normals
from settings import read_settings_file


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_compute_normals(device):
    read_settings_file()
    frame_data: StandaloneFrameDataset = StandaloneFramePreset.RED_SHORTS_200.value
    frame_data.load()
    depth_image = np.array(o3d.io.read_image(frame_data.get_depth_image_path()))
    intrinsics, _ = \
        data.camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(frame_data.get_intrinsics_path(),
                                                                          frame_data.get_depth_image_path())
    intrinsic_matrix = np.array(intrinsics.intrinsic_matrix)
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)
    dv = ((point_image[2:, :] - point_image[:-2, :])[:, 1:-1]).reshape(-1, 3)
    du = ((point_image[:, 2:] - point_image[:, :-2])[1:-1, :]).reshape(-1, 3)
    expected_normals = np.cross(du, dv)
    expected_normals /= np.tile(np.linalg.norm(expected_normals, axis=1).reshape(-1, 1), (1, 3))
    expected_normals[expected_normals[:, 2] > 0] = -expected_normals[expected_normals[:, 2] > 0]
    expected_normals = expected_normals.reshape(478, 638, -1)

    mask_y = np.logical_or(depth_image[2:, :] == 0, depth_image[:-2, :] == 0)[:, 1:-1]
    mask_x = np.logical_or(depth_image[:, 2:] == 0, depth_image[:, :-2] == 0)[1:-1, :]
    mask = np.logical_or(mask_x, mask_y)

    expected_normals[mask] = 0

    full_normals = compute_normals(device, point_image)
    full_normals_o3d = o3c.Tensor(full_normals)
    full_normals_o3d.save("/home/algomorph/Workbench/NeuralTracking/tests/test_data/red_shorts_200_normals.npy")

    normals = full_normals[1:479, 1:639]

    assert np.allclose(expected_normals, normals, atol=1e-7)
