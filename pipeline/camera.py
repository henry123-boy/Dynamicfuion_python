import numpy as np
import open3d as o3d
from typing import Tuple


def load_intrinsic_3x3_matrix_from_text_4x4_matrix(path_matrix: str) -> np.ndarray:
    intrinsic_matrix = np.loadtxt(path_matrix)
    return intrinsic_matrix[0:3, 0:3].copy()


def load_intrinsic_matrix_entries_from_text_4x4_matrix(path_matrix: str) -> Tuple[float, float, float, float]:
    intrinsic_matrix = np.loadtxt(path_matrix)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    return fx, fy, cx, cy


def load_intrinsic_matrix_entries_as_dict_from_text_4x4_matrix(path_matrix: str) -> dict:
    intrinsic_matrix = np.loadtxt(path_matrix)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }


def load_open3d_intrinsics_from_text_4x4_matrix_and_image(path_matrix: str, path_image: str) -> o3d.camera.PinholeCameraIntrinsic:
    intrinsic_matrix = np.loadtxt(path_matrix)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    image_shape = o3d.io.read_image(path_image).get_max_bound()
    width = int(image_shape[0])
    height = int(image_shape[1])
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def extract_intrinsic_projection_parameters(intrinsics: o3d.camera.PinholeCameraIntrinsic) -> Tuple[float, float, float, float]:
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    return fx, fy, cx, cy


def intrinsic_projection_parameters_as_dict(intrinsics: o3d.camera.PinholeCameraIntrinsic) -> dict:
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }
    return intrinsics


def print_intrinsic_projection_parameters(intrinsics: o3d.camera.PinholeCameraIntrinsic) -> None:
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    print("Intrinsics: \nfx={:f}\nfy={:f}\ncx={:f}\ncy={:f}".format(fx, fy, cx, cy))
