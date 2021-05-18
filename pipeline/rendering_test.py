# ==================================================================================================
# A minimal example that renders a triangle mesh into a depth image using predefined perspective projection matrix
# Copyright 2021 Gregory Kramida
#
# Please run script from repository root, i.e.:
# python3 ./pipeline/rendering_test.py
# ==================================================================================================
import sys
import open3d as o3d
import cv2
import numpy as np
import pytorch3d.renderer.cameras
import torch
import os

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings, MeshRenderer, SoftPhongShader, TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures.meshes import Meshes

import options
from data import camera

PROGRAM_EXIT_SUCCESS = 0


def main():
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(os.path.join(options.output_directory, "mesh_000000_red_shorts.ply"))
    depth_intrinsics_path = os.path.join(options.dataset_base_directory, "val/seq014/intrinsics.txt")

    torch_device = torch.device("cuda:0")

    vertices_numpy = np.array(mesh.vertices, dtype=np.float32)
    vertex_colors_numpy = np.ascontiguousarray(np.fliplr(np.array(mesh.vertex_colors, dtype=np.float32)))
    faces_numpy = np.array(mesh.triangles, dtype=np.int64)

    vertices_torch = torch.from_numpy(vertices_numpy).cuda().unsqueeze(0)
    vertices_rgb = torch.from_numpy(vertex_colors_numpy).cuda().unsqueeze(0)
    textures = TexturesVertex(verts_features=vertices_rgb)
    faces_torch = torch.from_numpy(faces_numpy).cuda().unsqueeze(0)

    meshes_torch3d = Meshes(vertices_torch, faces_torch, textures)

    camera_translation = torch.zeros((1, 3), dtype=torch.float32, device=torch_device)

    camera_rotation = torch.eye(3, dtype=torch.float32, device=torch_device).unsqueeze(0)

    lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                         specular_color=((0.0, 0.0, 0.0),), device=torch_device, location=[[0.0, 0.0, -3.0]])

    # region ===== COMPUTE CAMERA MATRIX =====
    fx_screen, fy_screen, px_screen, py_screen = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(depth_intrinsics_path)

    image_width = 640
    image_height = 480
    half_image_width = image_width // 2
    half_image_height = image_height // 2

    fx = fx_screen / half_image_height
    fy = fy_screen / half_image_height
    px = - (px_screen - half_image_width) / half_image_height
    py = - (py_screen - half_image_height) / half_image_height
    K = torch.tensor([[[fx, 0.0, px, 0.0],
                       [0.0, fy, py, 0.0],
                       [0.0, 0.0, 0.0, -1.0],
                       [0.0, 0.0, -1.0, 0.0]]], dtype=torch.float32)
    # endregion

    cameras: pytorch3d.renderer.cameras.PerspectiveCameras \
        = PerspectiveCameras(device=torch_device,
                             R=camera_rotation,
                             T=camera_translation,
                             K=K)

    rasterization_settings = RasterizationSettings(image_size=(480, 640),
                                                   cull_backfaces=True,
                                                   cull_to_frustum=True,
                                                   z_clip_value=0.5,
                                                   faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras, raster_settings=rasterization_settings)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=rasterization_settings
        ),
        shader=SoftPhongShader(
            device=torch_device,
            cameras=cameras,
            lights=lights
        )
    )

    fragments = rasterizer.forward(meshes_torch3d)
    z_buffer = fragments.zbuf.cpu().numpy().reshape(480, 640, 1)
    rendered_depth = z_buffer
    rendered_depth[rendered_depth == -1.0] = 0.0
    rendered_depth /= 4.0
    rendered_depth_uint8 = (rendered_depth * 255).astype(np.uint8)

    depth_image_path = os.path.join(options.dataset_base_directory, "val/seq014/depth/000000.png")
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000
    depth_image_uint8 = (depth_image * 255).astype(np.uint8)
    cv2.imshow("source depth", depth_image_uint8)
    cv2.waitKey()
    cv2.imwrite(os.path.join(options.output_directory, os.path.join(options.output_directory, "source_depth.png")), depth_image_uint8)

    images = renderer(meshes_torch3d)
    rendered_mesh = images[0, ..., :3].cpu().numpy()
    rendered_mesh_uint8 = (rendered_mesh * 255).astype(np.uint8)
    cv2.imshow("rendered mesh", rendered_mesh_uint8)
    cv2.waitKey()
    cv2.imwrite(os.path.join(options.output_directory, "rendered_mesh.png"), rendered_mesh_uint8)

    cv2.imshow("rendered depth", rendered_depth_uint8)
    cv2.waitKey()
    cv2.imwrite(os.path.join(options.output_directory, "rendered_depth.png"), rendered_depth_uint8)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
