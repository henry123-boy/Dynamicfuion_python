# ==================================================================================================
# A minimal example that renders a triangle mesh into a depth image using predefined perspective projection matrix
# Copyright 2021 Gregory Kramida
#
# Please run script from repository root, i.e.:
# python3 ./tsdf_management/rendering_test.py
# ==================================================================================================
import sys
import open3d as o3d
import cv2
import numpy as np
import os

from settings import settings_general
from data import camera
from rendering import ImageCamera
from rendering import CameraOpenGLColorRenderer

PROGRAM_EXIT_SUCCESS = 0


def main():
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(os.path.join(settings_general.output_directory, "mesh_000000_red_shorts.ply"))
    depth_intrinsics_path = os.path.join(settings_general.dataset_base_directory, "val/seq014/intrinsics.txt")

    vertices_numpy = np.array(mesh.vertices, dtype=np.float32)
    vertex_colors_numpy = np.fliplr(np.array(mesh.vertex_colors, dtype=np.float32)).copy()
    faces_numpy = np.array(mesh.triangles, dtype=np.int64)

    camera_translation = np.zeros((1, 3), dtype=np.float32)
    camera_rotation = np.eye(3, dtype=np.float32)
    # camera_rotation[1, 1] = -1
    # camera_rotation[2, 2] = -1

    # region ===== COMPUTE CAMERA MATRIX =====
    fx_screen, fy_screen, px_screen, py_screen = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(depth_intrinsics_path)

    image_width = 640
    image_height = 480

    camera_opengl = ImageCamera(fx_screen, fy_screen, px_screen, py_screen, image_width, image_height)
    # camera_opengl.set_intrinsic_matrix(intrinsic_matrix_3x3_numpy)
    camera_opengl.set_rotation_matrix(camera_rotation)
    camera_opengl.center = np.array([0, 0, 0])

    print(camera_opengl.get_intrinsic_matrix())
    print(camera_opengl.get_extrinsic_matrix())

    renderer = CameraOpenGLColorRenderer(image_width, image_height)
    renderer.set_camera(camera_opengl)
    renderer.set_mesh(vertices_numpy, faces_numpy, vertex_colors_numpy, faces_numpy)
    renderer.display()
    rendered_mesh = renderer.get_color(0)
    rendered_mesh = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGBA2BGRA)
    rendered_mesh_uint8 = (rendered_mesh * 255).astype(np.uint8)
    cv2.imshow("rendered mesh", rendered_mesh_uint8)
    cv2.waitKey()

    # fragments = rasterizer.forward(meshes_torch3d)
    # z_buffer = fragments.zbuf.cpu().numpy().reshape(480, 640, 1)
    # rendered_depth = z_buffer
    # rendered_depth[rendered_depth == -1.0] = 0.0
    # rendered_depth /= 4.0
    # rendered_depth_uint8 = (rendered_depth * 255).astype(np.uint8)
    #
    depth_image_path = os.path.join(settings_general.dataset_base_directory, "val/seq014/depth/000000.png")
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000
    depth_image_uint8 = (depth_image * 255).astype(np.uint8)
    cv2.imshow("source depth", depth_image_uint8)
    cv2.waitKey()
    cv2.imwrite(os.path.join(settings_general.output_directory, os.path.join(settings_general.output_directory, "source_depth.png")), depth_image_uint8)
    #
    # images = renderer(meshes_torch3d)
    # rendered_mesh = images[0, ..., :3].cpu().numpy()
    # rendered_mesh_uint8 = (rendered_mesh * 255).astype(np.uint8)
    # cv2.imshow("rendered mesh", rendered_mesh_uint8)
    # cv2.waitKey()
    # cv2.imwrite(os.path.join(options.output_directory, "rendered_mesh.png"), rendered_mesh_uint8)
    #
    # cv2.imshow("rendered depth", rendered_depth_uint8)
    # cv2.waitKey()
    # cv2.imwrite(os.path.join(options.output_directory, "rendered_depth.png"), rendered_depth_uint8)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
