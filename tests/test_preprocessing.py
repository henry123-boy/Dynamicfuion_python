import numpy as np
import open3d as o3d
from sklearn.preprocessing import normalize

import nnrt
import data.camera
from data import StandaloneFrameDataset, StandaloneFramePreset
from image_processing.numba_cuda.preprocessing import cuda_compute_normal


def test_compute_normals():
    frame_data: StandaloneFrameDataset = StandaloneFramePreset.RED_SHORTS_200.value
    depth_image = np.array(o3d.io.read_image(frame_data.get_depth_image_path()))
    intrinsics, _ = \
        data.camera.load_open3d_intrinsics_from_text_4x4_matrix_and_image(frame_data.get_intrinsics_path(),
                                                                          frame_data.get_depth_image_path())
    intrinsic_matrix = np.array(intrinsics.intrinsic_matrix)
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)
    dv = ((point_image[2:, :] - point_image[:-2, :])[:, 1:-1]).reshape(-1, 3)
    du = ((point_image[:, 2:] - point_image[:, :-2])[1:-1, :]).reshape(-1, 3)
    expected_normals = normalize(np.cross(du, dv), axis=1)
    expected_normals[expected_normals[:, 2] > 0] = -expected_normals[expected_normals[:, 2] > 0]
    expected_normals = expected_normals.reshape(478, 638, -1)

    mask_y = np.logical_or(depth_image[2:, :] == 0, depth_image[:-2, :] == 0)[:, 1:-1]
    mask_x = np.logical_or(depth_image[:, 2:] == 0, depth_image[:, :-2] == 0)[1:-1, :]
    mask = np.logical_or(mask_x, mask_y)

    expected_normals[mask] = 0

    normals = cuda_compute_normal(point_image)[1:479, 1:639]

    assert np.allclose(expected_normals, normals, atol=1e-7)