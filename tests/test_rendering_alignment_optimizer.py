from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
import nnrt
import torch.utils.dlpack as torch_dlpack

# code being tested
from alignment.render_based.rendering_alignment_optimizer import PureTorchOptimizer
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode
from data.camera import load_intrinsic_3x3_matrix_from_text_4x4_matrix


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
    subdivision_count = 2
    mesh_legacy: o3d.geometry.TriangleMesh = \
        o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).subdivide_midpoint(subdivision_count)
    mesh_legacy.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.7, 0.7, 0.7]] * len(mesh_legacy.vertices)))
    mesh: o3d.t.geometry.TriangleMesh = \
        o3d.t.geometry.TriangleMesh.from_legacy(
            mesh_legacy,
            vertex_dtype=o3c.float32, device=device
        )
    nnrt.geometry.compute_vertex_normals(mesh, True)
    box_center_position = o3c.Tensor([-0.5, -0.5, 2.5], dtype=o3c.float32, device=device)

    mesh.vertex["positions"] = mesh.vertex["positions"] + box_center_position
    mesh.vertex["colors"] = o3c.Tensor([[0.7, 0.7, 0.7]] * len(mesh.vertex["positions"]), dtype=o3c.float32,
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
        assert image_depth_gt.as_tensor().allclose(image_depth_o3d.as_tensor())
