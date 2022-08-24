import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytorch3d.structures.meshes
import torch
import torch.utils.dlpack as torch_dlpack
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures.meshes import Meshes

from typing import Tuple, Union


def open3d_mesh_to_pytorch3d(mesh: o3d.t.geometry.TriangleMesh) -> pytorch3d.structures.meshes.Meshes:
    vertices_o3d = mesh.vertex["positions"]

    faces_o3d = mesh.triangle["indices"]
    vertices_torch = torch_dlpack.from_dlpack(vertices_o3d.to_dlpack()).unsqueeze(0)
    faces_torch = torch_dlpack.from_dlpack(faces_o3d.to_dlpack()).unsqueeze(0)

    textures_torch = None
    if "colors" in mesh.vertex:
        vertices_rgb_o3d = mesh.vertex["colors"]
        vertices_rgb_torch = torch_dlpack.from_dlpack(vertices_rgb_o3d.to_dlpack()).unsqueeze(0)
        textures_torch = TexturesVertex(verts_features=vertices_rgb_torch)

    vertex_normals_torch = None
    if "normals" in mesh.vertex:
        vertex_normals_o3d = mesh.vertex["normals"]
        vertex_normals_torch = torch_dlpack.from_dlpack(vertex_normals_o3d.to_dlpack()).unsqueeze(0)

    return Meshes(vertices_torch, faces_torch, textures=textures_torch, verts_normals=vertex_normals_torch)


def make_ndc_intrinsic_matrix(image_size: Tuple[int, int], intrinsic_matrix: np.ndarray,
                              torch_device: torch.device) -> torch.Tensor:
    """
    Makes an intrinsic matrix in NDC (normalized device coordinates) coordinate system
    :param image_size: size of the output image, (height, width)
    :param intrinsic_matrix: 3x3 or 4x4 projection matrix of the camera
    :param torch_device: device on which to initialize the intrinsic matrix
    :return:
    """

    image_height, image_width = image_size
    half_image_width = image_width // 2
    half_image_height = image_height // 2

    fx_screen = intrinsic_matrix[0, 0]
    fy_screen = intrinsic_matrix[1, 1]
    px_screen = intrinsic_matrix[0, 2]
    py_screen = intrinsic_matrix[1, 2]

    fx = fx_screen / half_image_height
    fy = fy_screen / half_image_height
    px = -(px_screen - half_image_width) / half_image_height
    py = -(py_screen - half_image_height) / half_image_height
    # TODO due to what looks like a PyTorch3D bug, we have to use the 1.0 residuals here, not the below commented code
    #  residuals, and then use the non-identity rotation matrix...
    ndc_intrinsic_matrix = torch.tensor([[[fx, 0.0, px, 0.0],
                                          [0.0, fy, py, 0.0],
                                          [0.0, 0.0, 0.0, 1.0],
                                          [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32, device=torch_device)
    # ndc_intrinsic_matrix = torch.tensor([[[fx, 0.0, px, 0.0],
    #                                       [0.0, fy, py, 0.0],
    #                                       [0.0, 0.0, 0.0, -1.0],
    #                                       [0.0, 0.0, -1.0, 0.0]]], dtype=torch.float32, device=torch_device)
    return ndc_intrinsic_matrix


def build_pytorch3d_cameras_from_ndc(ndc_intrinsic_matrix: torch.Tensor,
                                     extrinsic_matrix: o3c.Tensor, torch_device: torch.device):
    extrinsics_torch: torch.Tensor = torch_dlpack.from_dlpack(extrinsic_matrix.to_dlpack())
    camera_rotation = (extrinsics_torch[:3, :3]).unsqueeze(0)
    camera_rotation *= torch.Tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]], dtype=np.float32)
    camera_translation = (extrinsics_torch[:3, 3]).reshape((1, 3)).unsqueeze(0)

    # when given extrinsics, reconstruct the camera
    cameras = PerspectiveCameras(device=torch_device,
                                 R=camera_rotation,
                                 T=camera_translation,
                                 K=ndc_intrinsic_matrix)
    return cameras


def build_pytorch3d_cameras(image_size: tuple, intrinsic_matrix: o3c.Tensor,
                            extrinsic_matrix: o3c.Tensor, device: Union[o3c.Device, torch.device]) \
        -> PerspectiveCameras:
    if type(device) == o3c.Device:
        if device.get_type() == o3c.Device.DeviceType.CUDA:
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device("cpu:0")
    else:
        torch_device = device

    ndc_intrinsic_matrix = make_ndc_intrinsic_matrix(image_size, intrinsic_matrix.cpu().numpy(), torch_device)
    return build_pytorch3d_cameras_from_ndc(ndc_intrinsic_matrix, extrinsic_matrix, torch_device)
