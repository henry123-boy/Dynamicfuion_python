from typing import Tuple, Union

import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack
import open3d as o3d
import open3d.core as o3c
from pytorch3d.renderer import BlendParams

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings, SoftPhongShader, TexturesVertex, MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures.meshes import Meshes

import rendering.converters as converters


class RenderMaskCode:
    DEPTH = 0b10
    RGB = 0b01


class PyTorch3DRenderer:
    def __init__(self, image_size: Tuple[int, int], device: o3c.Device, intrinsic_matrix: o3c.Tensor):
        """
        Construct a renderer to render to the specified size using the specified device and projective camera intrinsics.
        :param image_size: tuple (height, width) for the rendered image size
        :param device: the device to use for rendering
        :param intrinsic_matrix: a 3x3 or 4x4 intrinsics tensor
        """
        self.torch_device = converters.device_open3d_to_pytorch(device)
        self.lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                                  specular_color=((0.0, 0.0, 0.0),), device=self.torch_device,
                                  location=[[0.0, 0.0, -3.0]])

        self.K = converters.make_pytorch3d_ndc_intrinsic_matrix(image_size, intrinsic_matrix.cpu().numpy(),
                                                                self.torch_device)
        # FIXME (see comments in tsdf_management/subprocedure_examples/pytorch3d_rendering_test.py)
        # camera_rotation = (torch.eye(3, dtype=torch.float32, device=self.torch_device)).unsqueeze(0)
        camera_rotation = torch.tensor([[[-1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, 1]]], dtype=torch.float32, device=self.torch_device)
        self.cameras: PerspectiveCameras \
            = PerspectiveCameras(device=self.torch_device,
                                 R=camera_rotation,
                                 T=torch.zeros((1, 3), dtype=torch.float32, device=self.torch_device),
                                 K=self.K)

        self.rasterization_settings = RasterizationSettings(image_size=image_size,
                                                            cull_backfaces=False,
                                                            cull_to_frustum=True,
                                                            z_clip_value=0.5,
                                                            faces_per_pixel=1)

        self.rasterizer = MeshRasterizer(self.cameras, raster_settings=self.rasterization_settings)
        self.image_size = image_size

        self.shader = SoftPhongShader(
            device=self.torch_device,
            cameras=self.cameras,
            lights=self.lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def _set_up_rasterizer_and_renderer(self, extrinsics: Union[o3c.Tensor, None] = None):
        if extrinsics is not None:
            self.cameras = converters.build_pytorch3d_cameras_from_ndc(self.K, extrinsics, self.torch_device)

            self.rasterizer = MeshRasterizer(self.cameras, raster_settings=self.rasterization_settings)

            self.shader = SoftPhongShader(
                device=self.torch_device,
                cameras=self.cameras,
                lights=self.lights
            )

    def render_mesh_legacy(self, mesh: o3d.geometry.TriangleMesh,
                           extrinsics: Union[o3c.Tensor, None] = None,
                           depth_scale=1000.0) -> Tuple[np.ndarray, np.ndarray]:
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

        meshes_torch3d = Meshes(vertices_torch, faces_torch, textures=textures, verts_normals=None)

        self._set_up_rasterizer_and_renderer(extrinsics)

        fragments = self.rasterizer.forward(meshes_torch3d)
        rendered_depth = fragments.zbuf.cpu().numpy().reshape(self.image_size[0], self.image_size[1])
        rendered_depth[rendered_depth == -1.0] = 0.0
        rendered_depth *= depth_scale

        images = self.shader(fragments, meshes_torch3d)
        rendered_color = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return rendered_depth, rendered_color

    def render_mesh(self, mesh: o3d.t.geometry.TriangleMesh,
                    extrinsics: Union[o3c.Tensor, None] = None,
                    depth_scale=1000.0, render_mode_mask: int = RenderMaskCode.DEPTH | RenderMaskCode.RGB) \
            -> Tuple[Union[None, torch.Tensor], Union[None, torch.Tensor]]:
        """
        Render mesh to depth & color images compatible with typical RGB-D input depth & rgb images
        If the extrinsics matrix is provided, camera extrinsics are also updated for all subsequent renderings.
        Otherwise, the previous extrinsics are used. If no extrinsics were ever specified, uses an identity transform.
        :param mesh: the mesh to render
        :param extrinsics: an optional 4x4 camera transformation matrix.
        :param depth_scale: factor to scale depth (meters) by, commonly 1,000 in off-the-shelf RGB-D sensors
        :param render_mode_mask: bitwise mask that specifies which passes to render (See RenderMaskCode).
        :return:
        """
        meshes_torch3d = converters.open3d_mesh_to_pytorch3d(mesh)

        self._set_up_rasterizer_and_renderer(extrinsics)

        rendered_depth = None
        rendered_color = None

        fragments = self.rasterizer(meshes_torch3d)

        if render_mode_mask & RenderMaskCode.DEPTH > 0:
            rendered_depth = fragments.zbuf.clone().reshape(self.image_size[0], self.image_size[1])
            rendered_depth[rendered_depth == -1.0] = 0.0
            rendered_depth *= depth_scale
            rendered_depth = rendered_depth.to(torch.int16)

        if render_mode_mask & RenderMaskCode.RGB > 0:
            fragments = self.rasterizer(meshes_torch3d)
            images = self.shader(fragments, meshes_torch3d)
            rendered_color = (images[0, ..., :3] * 255).to(torch.uint8)
        return rendered_depth, rendered_color
