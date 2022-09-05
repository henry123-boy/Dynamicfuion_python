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

from alignment.render_based.pure_torch_render_based_optimizer import PureTorchRenderBasedOptimizer
# code being tested
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode
import rendering.converters as converters
from alignment.render_based.functional.warp_meshes import warp_meshes_using_node_anchors
from data.camera import load_intrinsic_3x3_matrix_from_text_4x4_matrix


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


@pytest.mark.parametrize("device", [o3c.Device('cuda:0'), o3c.Device('cpu:0')])
def test_pytorch3d_renderer(device):
    save_gt_data = False
    test_path = Path(__file__).parent.resolve()
    test_data_path = test_path / "test_data"
    intrinsics_test_data_path = test_data_path / "intrinsics"
    red_shorts_intrinsics = o3c.Tensor(
        load_intrinsic_3x3_matrix_from_text_4x4_matrix(str(intrinsics_test_data_path / "red_shorts_intrinsics.txt"))
    )

    renderer = PyTorch3DRenderer((480, 640), device, intrinsic_matrix=red_shorts_intrinsics.to(device))
    mesh = generate_test_box(box_side_length=1.0, box_center_position=(0.0, 0.0, 2.0), subdivision_count=2,
                             device=device)

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


def twist_box_corners_around_y_axis(corners: torch.Tensor, angle_degrees: float, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    top_rotation_matrix = torch.tensor(rotation_around_y_axis(angle_degrees), dtype=torch.float32, device=device)
    bottom_rotation_matrix = top_rotation_matrix.T

    corners_center = corners.mean(axis=0)
    # make corner nodes spin same direction as rotation (left-multiply requires us to flip the matrices)
    rotated_corners = corners_center + torch.cat(
        ((corners[:4, :] - corners_center).mm(bottom_rotation_matrix),
         (corners[4:, :] - corners_center).mm(top_rotation_matrix)), dim=0)

    corner_translations = rotated_corners - corners
    corner_rotations = torch.stack([top_rotation_matrix] * 4 + [bottom_rotation_matrix] * 4).to(device)
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
    box_center = (0.5, 0.5, 0.5)
    box_side_length = 1.0
    mesh_o3d = generate_test_box(box_side_length=box_side_length, box_center_position=box_center,
                                 subdivision_count=subdivision_count,
                                 device=converters.device_pytorch_to_open3d(device))
    meshes_torch = converters.open3d_mesh_to_pytorch3d(mesh_o3d)

    # place a node in every corner of the box
    nodes = compute_box_corners(box_side_length, box_center, device)
    nodes += torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=device)  # to make distances unique
    nodes_o3d = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(nodes))
    node_rotations, node_translations = twist_box_corners_around_y_axis(nodes, 22.5, device)

    anchor_count = 4
    node_coverage = 0.5
    anchors, weights = nnrt.geometry.compute_anchors_and_weights_euclidean(mesh_o3d.vertex["positions"],
                                                                           nodes_o3d, anchor_count, 0,
                                                                           node_coverage=node_coverage)

    vertex_anchors = torch_dlpack.from_dlpack(anchors.to_dlpack())
    vertex_anchor_weights = torch_dlpack.from_dlpack(weights.to_dlpack())

    meshes_warped = warp_meshes_using_node_anchors(meshes_torch, nodes, node_rotations, node_translations,
                                                   vertex_anchors,
                                                   vertex_anchor_weights)

    gt_device = ground_truth_vertices_torch.to(device)
    assert gt_device.allclose(meshes_warped.verts_packed())


def stretch_depth_image(depth_torch: torch.Tensor) -> torch.Tensor:
    depth_torch_normalized = depth_torch.clone().to(torch.float32)
    depth_max = depth_torch_normalized.max()
    depth_torch_normalized[depth_torch == 0] = 10000.0
    depth_min = depth_torch_normalized.min()
    depth_torch_normalized[depth_torch == 0] = depth_min
    depth_torch_normalized *= 256.0
    depth_torch_normalized /= (depth_max - depth_min)

    return depth_torch_normalized


@pytest.mark.parametrize("device", [o3c.Device('cuda:0'), o3c.Device('cpu:0')])
def test_loss_from_inputs(device: o3c.Device):
    save_images = True
    save_debug_images = True
    save_ground_truth = False
    mesh_o3d = generate_test_xy_plane(1.0, (0.0, 0.0, 2.0), subdivision_count=0, device=device)
    # place graph nodes at plane corners. They'll correspond to vertices as long as there is no subdivision.
    graph_nodes = mesh_o3d.vertex["positions"].clone()
    graph_edges = o3c.Tensor([[1, 2, -1, -1],
                              [0, 3, -1, -1],
                              [0, 3, -1, -1],
                              [1, 2, -1, -1]], dtype=o3c.int32, device=device)
    graph_edge_weights = o3c.Tensor([[0.5, 0.5, 0, 0],
                                     [0.5, 0.5, 0, 0],
                                     [0.5, 0.5, 0, 0],
                                     [0.5, 0.5, 0, 0]], dtype=o3c.float32, device=device)
    graph_clusters = o3c.Tensor([0, 0, 0, 0], dtype=o3c.int32, device=device)
    node_coverage = 0.5
    warp_field = \
        nnrt.geometry.GraphWarpField(graph_nodes, graph_edges, graph_edge_weights, graph_clusters,
                                     node_coverage, False, 4, 0)

    image_size = (480, 640)
    intrinsic_matrix = o3c.Tensor([[580., 0., 320.],
                                   [0., 580., 240.],
                                   [0., 0., 1.0]], dtype=o3c.float64, device=o3c.Device('cpu:0'))
    extrinsic_matrix = o3c.Tensor.eye(4, dtype=o3c.float64, device=o3c.Device('cpu:0'))
    renderer = PyTorch3DRenderer(image_size, device, intrinsic_matrix=intrinsic_matrix)
    test_path = Path(__file__).parent.resolve()

    test_data_path = test_path / "test_data"
    image_test_data_path = test_data_path / "images"
    tensor_test_data_path = test_data_path / "tensors"

    rotation_angle_y = 10
    rotation_matrix_y = o3c.Tensor(rotation_around_y_axis(rotation_angle_y), dtype=o3c.float32, device=device)
    mesh_rotated = rotate_mesh(mesh_o3d, rotation_matrix_y)
    depth_torch, color_torch = renderer.render_mesh(mesh_rotated)
    reference_image_depth_o3d = \
        o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(depth_torch)).to(o3c.uint16))
    reference_image_color_o3d = \
        o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(color_torch)))

    if save_images:
        reference_depth_path = image_test_data_path / f"plane_xy_rotated_y_{rotation_angle_y}_depth_000.png"
        reference_color_path = image_test_data_path / f"plane_xy_rotated_y_{rotation_angle_y}_color_000.png"
        o3d.t.io.write_image(str(reference_depth_path), reference_image_depth_o3d)
        o3d.t.io.write_image(str(reference_color_path), reference_image_color_o3d)
        if save_debug_images:
            reference_depth_normalized_path = \
                image_test_data_path / f"plane_xy_rotated_y_{rotation_angle_y}_depth_normalized_000.png"
            depth_torch_normalized = stretch_depth_image(depth_torch)
            reference_image_depth_normalized_o3d = o3d.t.geometry.Image(
                o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(depth_torch_normalized)).to(o3c.uint8))
            o3d.t.io.write_image(str(reference_depth_normalized_path), reference_image_depth_normalized_o3d)

    reference_points, reference_in_depth_range_mask = \
        nnrt.geometry.unproject_3d_points_without_depth_filtering(reference_image_depth_o3d, intrinsic_matrix,
                                                                  extrinsic_matrix, depth_scale=1000, depth_max=10.0,
                                                                  preserve_pixel_layout=False)
    reference_point_cloud = o3d.t.geometry.PointCloud(reference_points)

    optimizer = PureTorchRenderBasedOptimizer(reference_image_color_o3d, reference_point_cloud,
                                              reference_in_depth_range_mask.logical_not(),
                                              mesh_o3d, warp_field, intrinsic_matrix, extrinsic_matrix)

    residuals = \
        optimizer.compute_residuals_from_inputs(optimizer.graph_node_rotations, optimizer.graph_node_translations)

    # maximum distance has to be within some tolerance of [sin(angle) * hypotenuse], hypotenuse is half the side
    # length of the plane here.
    assert math.isclose(float(torch.sqrt(residuals.max()).cpu()), math.sin(math.radians(10.0)) * 0.5, abs_tol=1e-3)

    ground_truth_data_path = tensor_test_data_path / "render_based_icp_data_residual_ground_truth.pt"
    if save_ground_truth:
        torch.save(residuals, ground_truth_data_path)
        bad_indices = None
    else:
        ground_truth_residuals = torch.load(ground_truth_data_path).to(converters.device_open3d_to_pytorch(device))
        close = residuals.isclose(ground_truth_residuals, rtol=1.0, atol=1e-4)

        # we only expect some discrepancy along the diagonal, where PyTorch3D rendering tends to give inconsistent
        # results between CPU & CUDA
        bad_indices = torch.where(torch.logical_not(close))[0].cpu().numpy()
        bad_pixels = np.dstack(np.unravel_index(bad_indices, image_size))[0]
        bad_pixel_points = bad_pixels.astype(float)
        diagonal_start_point = np.array([103.0, 456.0])
        diagonal_end_point = np.array([382.0, 177.0])
        diagonal_direction = diagonal_end_point - diagonal_start_point
        diagonal_direction /= np.linalg.norm(diagonal_direction)
        closest_diagonal_stops = (
                bad_pixel_points.dot(diagonal_direction) - diagonal_start_point.dot(diagonal_direction))
        closest_diagonal_points = diagonal_start_point + np.repeat(closest_diagonal_stops.reshape(-1, 1), 2,
                                                                   axis=1) * diagonal_direction
        distances = np.linalg.norm(bad_pixel_points - closest_diagonal_points, axis=1)
        assert np.allclose(distances, np.zeros_like(distances))


    if save_images:
        normalized_residuals = residuals + residuals.min()
        normalized_residuals /= normalized_residuals.max()
        residuals_o3d_image_shape = \
            o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(normalized_residuals.reshape(image_size)))
        residuals_o3d_uint16: o3c.Tensor = (residuals_o3d_image_shape * 65535.0).to(o3c.uint16)
        rendered_depth, rendered_color = optimizer.render_warped_mesh()
        residuals_image = o3d.t.geometry.Image(residuals_o3d_uint16)
        residual_image_path = image_test_data_path / f"plane_xy_rotated_y_{rotation_angle_y}_residuals_000.png"
        o3d.t.io.write_image(str(residual_image_path), residuals_image)

        if save_debug_images and bad_indices is not None:
            residuals_color = (torch.tile(normalized_residuals.reshape(-1, 1), (1, 3)))
            bad_indices_torch = torch.tensor(bad_indices, device=converters.device_open3d_to_pytorch(device))
            residuals_color[bad_indices_torch] = \
                torch.tensor((1.0, 0, 0), device=converters.device_open3d_to_pytorch(device))

            residuals_color_uint8_image_shape = \
                (residuals_color * 255.0).to(torch.uint8).reshape(image_size[0], image_size[1], 3)
            residuals_debug_image = \
                o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(residuals_color_uint8_image_shape)))
            residual_debug_image_path = \
                image_test_data_path / f"plane_xy_rotated_y_{rotation_angle_y}_residuals_debug_000.png"
            o3d.t.io.write_image(str(residual_debug_image_path), residuals_debug_image)

        source_depth_o3d = \
            o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rendered_depth)).to(o3c.uint16))
        source_color_o3d = \
            o3d.t.geometry.Image(o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rendered_color)))
        source_color_path = image_test_data_path / "plane_xy_color_000.png"
        source_depth_path = image_test_data_path / "plane_xy_depth_000.png"
        o3d.t.io.write_image(str(source_depth_path), source_depth_o3d)
        o3d.t.io.write_image(str(source_color_path), source_color_o3d)
