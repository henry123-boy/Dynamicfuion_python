import sys
import cv2
import numpy as np
import torch
import torch.utils
import os
import typing

import open3d as o3d
import open3d.core as o3c

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings, MeshRenderer, SoftPhongShader, TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures.meshes import Meshes


def make_ndc_intrinsic_matrix(image_size: typing.Tuple[int, int], intrinsic_matrix: np.ndarray) -> torch.Tensor:
    """
    Makes an intrinsic matrix in NDC (normalized device coordinates) coordinate system
    :param image_size: size of the output image, (height, width)
    :param intrinsic_matrix: 3x3 or 4x4 projection matrix of the camera
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

    ndc_intrinsic_matrix = torch.tensor([[[fx, 0.0, px, 0.0],
                                          [0.0, fy, py, 0.0],
                                          [0.0, 0.0, 0.0, -1.0],
                                          [0.0, 0.0, -1.0, 0.0]]], dtype=torch.float32)
    return ndc_intrinsic_matrix


class Renderer:
    def __init__(self, image_size: typing.Tuple[int, int], device: o3c.Device, intrinsic_matrix: o3c.Tensor):
        """
        Construct a renderer to render to the specified size using the specified device and projective camera intrinsics.
        :param image_size: tuple (height, width) for the rendered image size
        :param device: the device to use for rendering
        :param intrinsic_matrix: a 3x3 or 4x4 intrinsics tensor
        """
        if device.DeviceType is o3c.Device.CUDA:
            self.torch_device = torch.device("cuda:0")
        else:
            self.torch_device = torch.device("cpu:0")
        self.lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                                  specular_color=((0.0, 0.0, 0.0),), device=self.torch_device, location=[[0.0, 0.0, -3.0]])

        self.K = make_ndc_intrinsic_matrix(image_size, intrinsic_matrix.cpu().numpy())
        self.cameras: PerspectiveCameras \
            = PerspectiveCameras(device=self.torch_device,
                                 R=(torch.eye(3, dtype=torch.float32)).unsqueeze(0),
                                 T=torch.zeros((1, 3), dtype=torch.float32, device=self.torch_device),
                                 K=self.K)

        self.rasterization_settings = RasterizationSettings(image_size=image_size,
                                                            cull_backfaces=True,
                                                            cull_to_frustum=True,
                                                            z_clip_value=0.5,
                                                            faces_per_pixel=1)

        self.rasterizer = MeshRasterizer(self.cameras, raster_settings=self.rasterization_settings)

        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=SoftPhongShader(
                device=self.torch_device,
                cameras=self.cameras,
                lights=self.lights
            )
        )

    def render_mesh(self, mesh: o3d.geometry.TriangleMesh,
                    extrinsics: typing.Union[o3c.Tensor, None] = None,
                    depth_scale=1000.0) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh to depth & color images compatible with typical RGB-D input depth & rgb images
        If the extrinsics matrix is provided, camera extrinsics are also updated for all subsequent renderings.
        Otherwise, the previous extrinsics are used. If no extrinsics were ever specified, uses an identity transform.
        :param mesh: the mesh to render
        :param extrinsics: an optional 4x4 camera transformation matrix.
        :param depth_scale: factor to scale depth (meters) by, commonly 1,000 in off-the-shelf RGB-D sensors
        :return:
        """
        vertices_numpy = np.array(mesh.vertices, dtype=np.float32)
        vertex_colors_numpy = np.fliplr(np.array(mesh.vertex_colors, dtype=np.float32)).copy()
        faces_numpy = np.array(mesh.triangles, dtype=np.int64)

        vertices_torch = torch.from_numpy(vertices_numpy).cuda().unsqueeze(0)
        vertices_rgb = torch.from_numpy(vertex_colors_numpy).cuda().unsqueeze(0)
        textures = TexturesVertex(verts_features=vertices_rgb)
        faces_torch = torch.from_numpy(faces_numpy).cuda().unsqueeze(0)

        meshes_torch3d = Meshes(vertices_torch, faces_torch, textures)

        extrinsics_torch: torch.Tensor = torch.utils.dlpack.from_dlpack(extrinsics.to_dlpack())
        camera_rotation = (extrinsics_torch[:3, :3]).unsqueeze(0)
        camera_translation = (extrinsics_torch[:3, 3]).reshape((1, 3)).unsqueeze(0)

        if extrinsics is not None:
            # when given extrinsics, reconstruct the camera
            self.cameras: PerspectiveCameras \
                = PerspectiveCameras(device=self.torch_device,
                                     R=camera_rotation,
                                     T=camera_translation,
                                     K=self.K)
            self.rasterizer = MeshRasterizer(self.cameras, raster_settings=self.rasterization_settings)

            self.renderer = MeshRenderer(
                rasterizer=self.rasterizer,
                shader=SoftPhongShader(
                    device=self.torch_device,
                    cameras=self.cameras,
                    lights=self.lights
                )
            )
        fragments = self.rasterizer.forward(meshes_torch3d)
        rendered_depth = fragments.zbuf.cpu().numpy().reshape(480, 640, 1)
        rendered_depth[rendered_depth == -1.0] = 0.0
        rendered_depth *= depth_scale

        images = self.renderer(meshes_torch3d)
        rendered_color = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return rendered_depth, rendered_color
