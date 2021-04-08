import math
import sys
import open3d as o3d
import cv2

import numpy as np
import pytorch3d.utils
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings, MeshRenderer, SoftPhongShader, TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures.meshes import Meshes

import torch
from pipeline import camera

PROGRAM_EXIT_SUCCESS = 0


def main():
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000000_red_shorts.ply")
    depth_intrinsics_path = "/mnt/Data/Reconstruction/real_data/deepdeform/v1_reduced/val/seq014/intrinsics.txt"
    # intrinsic_matrix = np.loadtxt(depth_intrinsics_path).astype(np.float32)
    fx, fy, cx, cy = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(depth_intrinsics_path)
    print(cx, cy)
    image_width = 640
    image_height = 480
    half_image_width = image_width // 2
    half_image_height = image_height // 2

    intrinsic_matrix = np.zeros((4, 4), dtype=np.float32)
    intrinsic_matrix[0, 0] = fx / half_image_height
    intrinsic_matrix[1, 1] = fy / half_image_height
    intrinsic_matrix[0, 2] = -(cx - half_image_width - 8) / half_image_width
    intrinsic_matrix[1, 2] = -(cy - half_image_height + 8) / half_image_height
    intrinsic_matrix[2, 2] = 0.0
    intrinsic_matrix[3, 3] = 0.0
    intrinsic_matrix[2, 3] = 1.0
    intrinsic_matrix[3, 2] = -1.0

    intrinsic_matrix_torch = torch.from_numpy(intrinsic_matrix).cuda().unsqueeze(0)

    torch_device = torch.device("cuda:0")

    vertices_numpy = np.array(mesh.vertices, dtype=np.float32)
    faces_numpy = np.array(mesh.triangles, dtype=np.int64)

    vertices_torch = torch.from_numpy(vertices_numpy).cuda().unsqueeze(0)
    vertices_rgb = torch.ones_like(vertices_torch)
    textures = TexturesVertex(verts_features=vertices_rgb)
    faces_torch = torch.from_numpy(faces_numpy).cuda().unsqueeze(0)

    meshes_torch3d = Meshes(vertices_torch, faces_torch, textures)

    camera_rotation = torch.eye(3, dtype=torch.float32, device=torch_device)[None]  # (1, 3, 3)
    camera_translation = torch.zeros(1, 3, dtype=torch.float32, device=torch_device)  # (1, 3)

    lights = PointLights(device=torch_device, location=[[0.0, 0.0, -3.0]])

    cameras = PerspectiveCameras(device=torch_device,
                                 R=camera_rotation,
                                 T=camera_translation,
                                 K=intrinsic_matrix_torch)

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

    zbuf = fragments.zbuf.cpu().numpy().reshape(480, 640, 1)
    rendered_depth = zbuf
    rendered_depth[rendered_depth == -1.0] = 0.0
    rendered_depth /= 4.0
    rendered_depth_uint8 = (rendered_depth * 255).astype(np.uint8)
    cv2.imshow("image", rendered_depth_uint8)
    cv2.waitKey()
    cv2.imwrite("rendered_depth.png", rendered_depth_uint8)

    images = renderer(meshes_torch3d)
    rendered_mesh = images[0, ..., :3].cpu().numpy()
    rendered_mesh_uint8 = (rendered_mesh * 255).astype(np.uint8)
    cv2.imshow("image", rendered_mesh_uint8)
    cv2.waitKey()
    cv2.imwrite("rendered_mesh.png", rendered_mesh_uint8)
    depth_image_path = "/mnt/Data/Reconstruction/real_data/deepdeform/v1_reduced/val/seq014/depth/000000.png"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000
    cv2.imshow("image", (depth_image * 255).astype(np.uint8))
    cv2.waitKey()
    cv2.imwrite("source_depth.png", (depth_image * 255).astype(np.uint8))
    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
