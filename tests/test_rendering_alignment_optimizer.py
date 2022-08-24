#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 8/23/22.
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
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
import nnrt
import torch
import torch.utils.dlpack as torch_dlpack

import pytorch3d.structures as p3ds
from nnrt.geometry import compute_anchors_and_weights_shortest_path, compute_anchors_and_weights_euclidean

# code being tested
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode
from rendering.converters import open3d_mesh_to_pytorch3d
from alignment.render_based.functional.warp_meshes import warp_meshes_using_node_anchors
from data.camera import load_intrinsic_3x3_matrix_from_text_4x4_matrix


def device_open3d_to_pytorch(device: o3c.Device) -> torch.device:
    return torch.device(str(device).lower())


def device_pytorch_to_open3d(device: torch.device) -> o3c.Device:
    return o3c.Device(str(device))


def generate_test_box(device: o3c.Device, box_side_length: float = 1.0,
                      box_center_position: tuple = (-0.5, -0.5, 2.5),
                      subdivision_count: int = 2) -> o3d.t.geometry.TriangleMesh:
    mesh_legacy: o3d.geometry.TriangleMesh = \
        o3d.geometry.TriangleMesh.create_box(box_side_length,
                                             box_side_length, box_side_length).subdivide_midpoint(subdivision_count)
    mesh: o3d.t.geometry.TriangleMesh = \
        o3d.t.geometry.TriangleMesh.from_legacy(
            mesh_legacy,
            vertex_dtype=o3c.float32, device=device
        )
    nnrt.geometry.compute_vertex_normals(mesh, True)
    box_center_position = o3c.Tensor(list(box_center_position), dtype=o3c.float32, device=device)

    mesh.vertex["positions"] = mesh.vertex["positions"] + box_center_position
    mesh.vertex["colors"] = o3c.Tensor([[0.7, 0.7, 0.7]] * len(mesh.vertex["positions"]), dtype=o3c.float32,
                                       device=device)
    return mesh


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_pytorch3d_renderer(device):
    save_gt_data = False
    test_path = Path(__file__).parent.resolve()
    test_data_path = test_path / "test_data"
    intrinsics_test_data_path = test_data_path / "intrinsics"
    red_shorts_intrinsics = o3c.Tensor(
        load_intrinsic_3x3_matrix_from_text_4x4_matrix(str(intrinsics_test_data_path / "red_shorts_intrinsics.txt"))
    )

    renderer = PyTorch3DRenderer((480, 640), device, intrinsic_matrix=red_shorts_intrinsics.to(device))
    mesh = generate_test_box(device)

    depth_torch, rgb_torch = renderer.render_mesh(mesh, render_mode_mask=RenderMaskCode.DEPTH | RenderMaskCode.RGB)

    image_depth_o3d = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(depth_torch)).to(o3c.uint16))
    image_rgb_o3d = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rgb_torch)))

    image_test_data_path = test_data_path / "images"
    image_rgb_gt_path = image_test_data_path / "box_color_000.png"
    image_depth_gt_path = image_test_data_path / "box_depth_000.png"

    if save_gt_data:
        o3d.t.io.write_image(str(image_rgb_gt_path), image_rgb_o3d)
        o3d.t.io.write_image(str(image_depth_gt_path), image_depth_o3d)
    else:
        image_rgb_gt = o3d.t.io.read_image(str(image_rgb_gt_path)).to(device)
        assert image_rgb_gt.as_tensor().allclose(image_rgb_o3d.as_tensor())
        image_depth_gt = o3d.t.io.read_image(str(image_depth_gt_path)).to(device)
        assert image_depth_gt.as_tensor().to(o3c.float32).allclose(image_depth_o3d.as_tensor().to(o3c.float32),
                                                                   rtol=1e-3, atol=1e-3)


def compute_box_corners(box_side_length: float, box_center_position: tuple, device: torch.device) -> torch.Tensor:
    slh = box_side_length / 2
    bc = box_center_position
    return torch.tensor([
        # around bottom (-y) face
        [bc[0] - slh, bc[1] - slh, bc[2] - slh],
        [bc[0] + slh, bc[1] - slh, bc[2] - slh],
        [bc[0] - slh, bc[1] - slh, bc[2] + slh],
        [bc[0] + slh, bc[1] - slh, bc[2] + slh],
        # around top (+y) face
        [bc[0] - slh, bc[1] + slh, bc[2] - slh],
        [bc[0] + slh, bc[1] + slh, bc[2] - slh],
        [bc[0] - slh, bc[1] + slh, bc[2] + slh],
        [bc[0] + slh, bc[1] + slh, bc[2] + slh],
    ], dtype=torch.float32, device=device)


def twist_box_corners_around_y_axis(corners: torch.Tensor, angle_degrees: float, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    mesh_rotation_angle = math.radians(angle_degrees)
    global_rotation_matrix_top = torch.tensor(
        [[math.cos(mesh_rotation_angle), 0.0, math.sin(mesh_rotation_angle)],
         [0., 1., 0.],
         [-math.sin(mesh_rotation_angle), 0.0, math.cos(mesh_rotation_angle)]],
        dtype=torch.float32, device=device
    )
    global_rotation_matrix_bottom = torch.tensor(
        [[math.cos(-mesh_rotation_angle), 0.0, math.sin(-mesh_rotation_angle)],
         [0., 1., 0.],
         [-math.sin(-mesh_rotation_angle), 0.0, math.cos(-mesh_rotation_angle)]],
        dtype=torch.float32, device=device
    )

    corners_center = corners.mean(axis=0)
    rotated_corners = corners_center + torch.cat(
        ((corners[:4, :] - corners_center).mm(global_rotation_matrix_bottom),
         (corners[4:, :] - corners_center).mm(global_rotation_matrix_top)), dim=0)
    corner_translations = rotated_corners - corners
    corner_rotations = torch.stack([global_rotation_matrix_top] * 4 + [global_rotation_matrix_bottom] * 4).to(device)
    return corner_rotations, corner_translations


@pytest.fixture(scope="module")
def ground_truth_vertices_torch() -> torch.Tensor:
    vertices = torch.tensor([[-2.29245767e-01, -1.41411602e-08, 2.67084956e-01],
                             [6.91970110e-01, -1.42059857e-08, -7.78945014e-02],
                             [1.16394386e-01, -1.43351269e-08, 1.19694209e+00],
                             [1.04608846e+00, -1.49011612e-08, 8.31702471e-01],
                             [2.23095283e-01, 1.00000012e+00, -7.21708238e-02],
                             [1.13396776e+00, 9.99999940e-01, 1.43104389e-01],
                             [4.56829136e-03, 9.99999940e-01, 8.61463904e-01],
                             [9.16629255e-01, 1.00000000e+00, 1.09062088e+00],
                             [5.91987669e-01, 1.00000012e+00, 4.94710982e-01],
                             [1.04602730e+00, 1.00000000e+00, 6.24803603e-01],
                             [6.72028065e-01, 9.99999940e-01, 4.03444134e-02],
                             [1.07328296e-01, 1.00000000e+00, 3.99524152e-01],
                             [4.31075335e-01, 1.00000000e+00, 9.46507633e-01],
                             [-5.95213622e-02, -1.42774095e-08, 7.33223557e-01],
                             [2.05226243e-03, 4.99999970e-01, 6.10076189e-01],
                             [-8.64362046e-02, 5.00000000e-01, 1.59977779e-01],
                             [7.47523531e-02, 5.00000000e-01, 1.07201600e+00],
                             [2.22478822e-01, -1.43897658e-08, 9.65380520e-02],
                             [3.92807037e-01, -1.49011612e-08, 5.61104476e-01],
                             [8.54746759e-01, -1.49011612e-08, 3.69762778e-01],
                             [5.84148765e-01, -1.49011612e-08, 1.02304411e+00],
                             [8.53231609e-01, 4.99999970e-01, 2.73631583e-03],
                             [9.07589078e-01, 4.99999940e-01, 4.40219164e-01],
                             [1.01334465e+00, 5.00000000e-01, 8.97190273e-01],
                             [5.57375371e-01, 5.00000000e-01, 1.00965750e+00],
                             [3.91291916e-01, 5.00000000e-01, 7.54364058e-02]],
                            dtype=torch.float32,
                            device=torch.device("cpu:0"))
    return vertices


@pytest.mark.parametrize("device", [torch.device('cuda:0'), torch.device('cpu:0')])
def test_warp_meshes_using_node_anchors(device: torch.device, ground_truth_vertices_torch):
    subdivision_count = 1
    box_center = (0.0, 0.0, 0.0)
    box_side_length = 2.0
    mesh_o3d = generate_test_box(device_pytorch_to_open3d(device), box_side_length=box_side_length,
                                 box_center_position=box_center,
                                 subdivision_count=subdivision_count)
    meshes_torch = open3d_mesh_to_pytorch3d(mesh_o3d)

    # place a node in every corner of the box
    nodes = compute_box_corners(box_side_length, box_center, device)
    nodes_o3d = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(nodes))
    node_rotations, node_translations = twist_box_corners_around_y_axis(nodes, 22.5, device)

    anchor_count = 4
    node_coverage = 0.5
    anchors, weights = compute_anchors_and_weights_euclidean(mesh_o3d.vertex["positions"],
                                                             nodes_o3d, anchor_count, 0,
                                                             node_coverage=node_coverage)

    vertex_anchors = torch_dlpack.from_dlpack(anchors.to_dlpack())
    vertex_anchor_weights = torch_dlpack.from_dlpack(weights.to_dlpack())

    meshes_warped = warp_meshes_using_node_anchors(meshes_torch, nodes, node_rotations, node_translations,
                                                   vertex_anchors,
                                                   vertex_anchor_weights)

    gt_device = ground_truth_vertices_torch.to(device)
    assert gt_device.allclose(meshes_warped.verts_packed())
