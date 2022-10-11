#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 9/19/22.
#  Copyright (c) 2022 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import math
import sys
from collections import namedtuple
from typing import Tuple, NamedTuple, Union
from enum import Enum

import cv2
import nnrt
import open3d as o3d
import open3d.core as o3c
import pytorch3d.renderer as p3dr
import pytorch3d.renderer.mesh.shader as p3dr_shader
from pytorch3d.renderer.mesh.rasterizer import Fragments
import torch
import torch.utils.dlpack as torch_dlpack
import numpy as np

import rendering.converters

GENERATED_TEST_DATA_DIR = "/home/algomorph/Workbench/NeuralTracking/cmake-build-debug/csrc/tests/test_data"
STATIC_TEST_DATA_DIR = "/home/algomorph/Workbench/NeuralTracking/csrc/tests/test_data"


def generate_test_xy_plane(plane_side_length: float, plane_center_position: tuple,
                           subdivision_count: int, device: o3c.Device) -> o3d.t.geometry.TriangleMesh:
    mesh = o3d.t.geometry.TriangleMesh(device=device)
    hsl = plane_side_length / 2.0
    mesh.vertex["positions"] = o3d.core.Tensor([[-hsl, -hsl, 0],  # bottom left
                                                [-hsl, hsl, 0],  # top left
                                                [hsl, -hsl, 0],  # bottom right
                                                [hsl, hsl, 0]],  # top right
                                               o3c.float32, device)
    mesh.triangle["indices"] = o3d.core.Tensor([[0, 1, 2],
                                                [2, 1, 3]], o3c.int64, device)
    mesh.triangle["normals"] = o3d.core.Tensor([[0, 0, -1],
                                                [0, 0, -1]], o3c.float32, device)
    mesh.vertex["normals"] = o3d.core.Tensor([[0, 0, -1],
                                              [0, 0, -1],
                                              [0, 0, -1],
                                              [0, 0, -1]], o3c.float32, device)

    mesh.vertex["colors"] = o3c.Tensor([[0.7, 0.7, 0.7]] * len(mesh.vertex["positions"]), dtype=o3c.float32,
                                       device=device)
    plane_center_position = o3c.Tensor(list(plane_center_position), dtype=o3c.float32, device=device)

    mesh.vertex["positions"] += plane_center_position

    if subdivision_count > 0:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh.to_legacy().subdivide_midpoint(subdivision_count),
                                                       vertex_dtype=o3c.float32, device=device)
    return mesh


def generate_test_box(box_side_length: float, box_center_position: tuple, subdivision_count: int,
                      device: o3c.Device) -> o3d.t.geometry.TriangleMesh:
    mesh_legacy: o3d.geometry.TriangleMesh = \
        o3d.geometry.TriangleMesh.create_box(box_side_length,
                                             box_side_length, box_side_length)
    if subdivision_count > 0:
        mesh_legacy = mesh_legacy.subdivide_midpoint(subdivision_count)
    mesh: o3d.t.geometry.TriangleMesh = \
        o3d.t.geometry.TriangleMesh.from_legacy(
            mesh_legacy,
            vertex_dtype=o3c.float32, device=device
        )
    nnrt.geometry.compute_vertex_normals(mesh, True)
    box_center_position = o3c.Tensor(list(box_center_position), dtype=o3c.float32, device=device)

    half_side_length = box_side_length / 2
    # Open3D doesn't generate boxes at the center of the coordinate grid -- rather, they are placed with one of the
    # corners in the origin. Thanks, Open3D. Not.
    box_center_initial_offset = o3c.Tensor([-half_side_length, -half_side_length, -half_side_length],
                                           dtype=o3c.float32, device=device)

    mesh.vertex["positions"] = mesh.vertex["positions"] + box_center_position + box_center_initial_offset

    mesh.vertex["colors"] = o3c.Tensor([[0.7, 0.7, 0.7]] * len(mesh.vertex["positions"]), dtype=o3c.float32,
                                       device=device)
    return mesh


def rotation_around_y_axis(angle_degrees: float) -> np.array:
    """
    # equivalent of MATLAB's roty. Damn you, MATLAB, for encouraging horrid coding techniques by severely abbreviating
    # function names.
    :param angle_degrees: angle, in degrees
    :return: matrix representation of the @param angle_degrees degree rotation (of a 3x1 vector) around y-axis.
    """
    angle_radians = math.radians(angle_degrees)
    return np.array([[math.cos(angle_radians), 0.0, math.sin(angle_radians)],
                     [0., 1., 0.],
                     [-math.sin(angle_radians), 0.0, math.cos(angle_radians)]])


def rotate_mesh(mesh: o3d.t.geometry.TriangleMesh, rotation: o3c.Tensor) -> o3d.t.geometry.TriangleMesh:
    """
    Rotates mesh around mean of its vertices using the specified rotation matrix. Does not modify input.
    :param mesh: mesh to rotate
    :param rotation: rotation matrix
    :return: rotated mesh
    """
    pivot = mesh.vertex["positions"].mean(dim=0)
    rotated_mesh = mesh.clone()
    rotated_vertices = (mesh.vertex["positions"] - pivot).matmul(rotation.T()) + pivot
    rotated_mesh.vertex["positions"] = rotated_vertices
    if "normals" in mesh.vertex:
        rotated_normals = mesh.vertex["normals"].matmul(rotation.T())
        rotated_mesh.vertex["normals"] = rotated_normals

    return rotated_mesh


def build_rasterizer_and_shader(image_size: Tuple[int, int], device: torch.device, no_shading: bool, shade_flat: bool,
                                light_position: Union[Tuple[float, float, float], None], naive: bool = True) -> \
        Tuple[p3dr.MeshRasterizer, p3dr_shader.ShaderBase]:
    if naive:
        rasterization_settings = p3dr.RasterizationSettings(image_size=image_size,
                                                            perspective_correct=False,
                                                            cull_backfaces=False,
                                                            cull_to_frustum=True,
                                                            z_clip_value=0.1,
                                                            faces_per_pixel=1,
                                                            bin_size=0,
                                                            max_faces_per_bin=0)
    else:
        rasterization_settings = p3dr.RasterizationSettings(image_size=image_size,
                                                            perspective_correct=False,
                                                            cull_backfaces=False,
                                                            cull_to_frustum=True,
                                                            z_clip_value=0.1,
                                                            faces_per_pixel=1)

    intrinsic_matrix_o3d = o3c.Tensor([[580., 0., 320.],
                                       [0., 580., 240.],
                                       [0., 0., 1.0]], dtype=o3c.float64, device=o3c.Device('cpu:0'))

    K = rendering.converters.make_pytorch3d_ndc_intrinsic_matrix(image_size, intrinsic_matrix_o3d.cpu().numpy(),
                                                                 device)
    camera_rotation = torch.tensor([[[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]]], dtype=torch.float32, device=device)
    cameras: p3dr.PerspectiveCameras \
        = p3dr.PerspectiveCameras(device=device,
                                  R=camera_rotation,
                                  T=torch.zeros((1, 3), dtype=torch.float32, device=device),
                                  K=K)
    if no_shading:
        lights = p3dr.PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                                  specular_color=((0.0, 0.0, 0.0),), device=device,
                                  location=[[0.0, 0.0, -3.0]])
    else:
        lights = p3dr.PointLights(ambient_color=((0.0, 0.0, 0.0),), diffuse_color=((0.5, 0.5, 0.5),),
                                  specular_color=((0.3, 0.3, 0.3),), device=device,
                                  location=[list(light_position)])
    rasterizer = p3dr.MeshRasterizer(cameras, raster_settings=rasterization_settings)
    if shade_flat:
        shader = p3dr.HardFlatShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=p3dr.BlendParams(background_color=(0.0, 0.0, 0.0))
        )
    else:
        shader = p3dr.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=p3dr.BlendParams(background_color=(0.0, 0.0, 0.0))
        )
    return rasterizer, shader


def make_test_data_rasterize_mesh(mesh: o3d.t.geometry.TriangleMesh, file_prefix: str, display_rendered: bool = False,
                                  no_shading: bool = True, shade_flat: bool = False,
                                  light_position: Union[Tuple[float, float, float], None] = None,
                                  naive: bool = True) -> None:
    image_size = (480, 640)
    device_o3d = mesh.vertex["positions"].device
    device_torch = rendering.converters.device_open3d_to_pytorch(device_o3d)
    mesh_torch = rendering.converters.open3d_mesh_to_pytorch3d(mesh)
    rasterizer, shader = \
        build_rasterizer_and_shader(image_size, device_torch, no_shading, shade_flat, light_position, naive)
    fragments = rasterizer(mesh_torch)
    rendered_color = shader(fragments, mesh_torch)
    rendered_color_uint8 = (rendered_color[0, ..., :3] * 255).to(torch.uint8)

    if display_rendered:
        cv2.imshow("rendered color", rendered_color_uint8.cpu().numpy())
        cv2.waitKey()
        rendered_color_o3d = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rendered_color_uint8))
        o3d.t.io.write_image(
            f"{GENERATED_TEST_DATA_DIR}/images/{file_prefix}_render_preview.png",
            o3d.t.geometry.Image(rendered_color_o3d)
        )

    np.save(
        f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_face_indices.npy",
        fragments.pix_to_face.cpu().numpy().reshape(image_size[0], image_size[1], 1))
    np.save(f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_depths.npy",
            fragments.zbuf.cpu().numpy().reshape(image_size[0], image_size[1], 1))
    np.save(
        f"{GENERATED_TEST_DATA_DIR}/"
        f"arrays/{file_prefix}_pixel_barycentric_coordinates.npy",
        fragments.bary_coords.cpu().numpy().reshape(image_size[0], image_size[1], 1, 3))
    np.save(
        f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_face_distances.npy",
        fragments.dists.cpu().numpy().reshape(image_size[0], image_size[1], 1))


def load_fragments(device_torch: torch.device, file_prefix: str) -> Fragments:
    pix_to_face = torch.tensor(np.load(
        f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_face_indices.npy")
    ).to(device_torch).unsqueeze(0)
    pixel_depths = torch.tensor(np.load(
        f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_depths.npy")
    ).to(device_torch).unsqueeze(0)
    bary_coords = torch.tensor(
        np.load(f"{GENERATED_TEST_DATA_DIR}/"
                f"arrays/{file_prefix}_pixel_barycentric_coordinates.npy")
    ).to(device_torch).unsqueeze(0)
    dists = torch.tensor(np.load(
        f"{GENERATED_TEST_DATA_DIR}/arrays/{file_prefix}_pixel_face_distances.npy")
    ).to(device_torch).unsqueeze(0)
    return Fragments(pix_to_face, pixel_depths, bary_coords, dists)


def shade_loaded_fragments(mesh: o3d.t.geometry.TriangleMesh, file_prefix: str, display_rendered: bool = False,
                           no_shading=True, shade_flat=False,
                           light_position: Union[Tuple[float, float, float], None] = None):
    image_size = (480, 640)
    device_o3d = mesh.vertex["positions"].device
    device_torch = rendering.converters.device_open3d_to_pytorch(device_o3d)

    fragments = load_fragments(device_torch, file_prefix=file_prefix)
    _, shader = build_rasterizer_and_shader(image_size, device_torch, no_shading, shade_flat, light_position)
    mesh_torch = rendering.converters.open3d_mesh_to_pytorch3d(mesh)
    rendered_color = shader(fragments, mesh_torch)
    rendered_color_uint8 = (rendered_color[0, ..., :3] * 255).to(torch.uint8)

    if display_rendered:
        cv2.imshow("rendered color", rendered_color_uint8.cpu().numpy())
        cv2.waitKey()
        rendered_color_o3d = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rendered_color_uint8))
        o3d.t.io.write_image(
            f"{GENERATED_TEST_DATA_DIR}/images/{file_prefix}_render_preview.png",
            o3d.t.geometry.Image(rendered_color_o3d)
        )


def load_mesh_paint_and_offset(mesh_path: str, offset: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                               color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                               device: o3c.Device = o3c.Device("CUDA:0")) \
        -> o3d.t.geometry.TriangleMesh:
    mesh = \
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(mesh_path),
            vertex_dtype=o3c.float32, triangle_dtype=o3c.int64, device=device
        )
    mesh.vertex["positions"] += o3c.Tensor(list(offset), dtype=o3c.float32, device=device)
    mesh.vertex["colors"] = o3c.Tensor([list(color)] * len(mesh.vertex["positions"]), dtype=o3c.float32, device=device)
    return mesh


PROGRAM_EXIT_SUCCESS = 0


class MeshDataSet(NamedTuple):
    prefix: str
    no_shading: bool
    shade_flat: bool
    light_position: Union[Tuple[float, float, float], None]


class MeshDataPreset(Enum):
    PLANE_0 = MeshDataSet("plane_0", no_shading=True, shade_flat=True, light_position=None)
    CUBE_0 = MeshDataSet("cube_0", no_shading=False, shade_flat=True, light_position=(1.0, 1.0, 3.0))
    BUNNY_RES4 = MeshDataSet("mesh_bunny_res4", False, False, (1.0, 1.0, 3.0))
    BUNNY_RES2 = MeshDataSet("mesh_bunny_res2", False, False, (1.0, 1.0, 3.0))
    M64_BUNNY_ARRAY = MeshDataSet("mesh_64_bunny_array", False, False, (1.0, 1.0, -3.0))


class Mode(Enum):
    GENERATE = 0,
    SHADE_LOADED = 1


def main():
    device_o3d = o3c.Device("CUDA:0")

    mesh_data_set = MeshDataPreset.BUNNY_RES4
    mode = Mode.SHADE_LOADED
    naive_rasterization = False

    get_mesh_by_data_set = {
        MeshDataPreset.PLANE_0: lambda: generate_test_xy_plane(1.0, (0.0, 0.0, 2.0), subdivision_count=0,
                                                               device=device_o3d),
        MeshDataPreset.CUBE_0: lambda: rotate_mesh(
            generate_test_box(1.0, (0.0, 0.0, 2.0), subdivision_count=0, device=device_o3d),
            o3c.Tensor(rotation_around_y_axis(45.0), device=device_o3d).to(o3c.float32)
        ),
        MeshDataPreset.BUNNY_RES4: lambda: load_mesh_paint_and_offset(
            "/mnt/Data/Reconstruction/real_data/Stanford/bunny/test_scenes/mesh_bunny_res4/"
            "meshes/mesh_bunny_res4.ply",
            offset=(0.0, -0.1, 0.3),
            device=device_o3d
        ),
        MeshDataPreset.BUNNY_RES2: lambda: load_mesh_paint_and_offset(
            "/mnt/Data/Reconstruction/real_data/Stanford/bunny/test_scenes/mesh_bunny_res2/"
            "meshes/mesh_bunny_res2.ply",
            offset=(0.0, -0.1, 0.3),
            device=device_o3d
        ),
        MeshDataPreset.M64_BUNNY_ARRAY: lambda: load_mesh_paint_and_offset(
            "/home/algomorph/Workbench/NeuralTracking/cmake-build-debug/csrc/tests/test_data/"
            "meshes/mesh_64_bunny_array.ply",
            device=device_o3d
        )
    }

    mesh = get_mesh_by_data_set[mesh_data_set]()
    if mode == Mode.GENERATE:
        make_test_data_rasterize_mesh(mesh, mesh_data_set.value.prefix, display_rendered=True,
                                      no_shading=mesh_data_set.value.no_shading,
                                      shade_flat=mesh_data_set.value.shade_flat,
                                      light_position=mesh_data_set.value.light_position,
                                      naive=naive_rasterization)
    elif mode == Mode.SHADE_LOADED:
        shade_loaded_fragments(mesh, mesh_data_set.value.prefix + "_out", True,
                               no_shading=mesh_data_set.value.no_shading,
                               shade_flat=mesh_data_set.value.shade_flat,
                               light_position=mesh_data_set.value.light_position)

    save_mesh_to_static_mesh_directory = False
    if save_mesh_to_static_mesh_directory:
        mesh_legacy: o3d.geometry.TriangleMesh = mesh.to_legacy()
        o3d.io.write_triangle_mesh(f"{STATIC_TEST_DATA_DIR}/meshes/" + mesh_data_set.value.prefix + ".ply", mesh_legacy)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
