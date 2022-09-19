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
import open3d as o3d
import open3d.core as o3c
import pytorch3d.renderer as p3dr
import torch
import numpy as np

import rendering.converters


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


if __name__ == "__main__":
    image_size = (480, 640)
    device_o3d = o3c.Device("CPU:0")
    device_torch = rendering.converters.device_open3d_to_pytorch(device_o3d)
    mesh_o3d = generate_test_xy_plane(1.0, (0.0, 0.0, 2.0), subdivision_count=0, device=device_o3d)
    mesh_torch = rendering.converters.open3d_mesh_to_pytorch3d(mesh_o3d)
    rasterization_settings = p3dr.RasterizationSettings(image_size=image_size,
                                                        perspective_correct=False,
                                                        cull_backfaces=False,
                                                        cull_to_frustum=True,
                                                        z_clip_value=0.1,
                                                        faces_per_pixel=1,
                                                        bin_size=0,
                                                        max_faces_per_bin=0)

    intrinsic_matrix_o3d = o3c.Tensor([[580., 0., 320.],
                                       [0., 580., 240.],
                                       [0., 0., 1.0]], dtype=o3c.float64, device=o3c.Device('cpu:0'))

    K = rendering.converters.make_pytorch3d_ndc_intrinsic_matrix(image_size, intrinsic_matrix_o3d.cpu().numpy(),
                                                                 device_torch)
    camera_rotation = torch.tensor([[[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]]], dtype=torch.float32, device=device_torch)
    cameras: p3dr.PerspectiveCameras \
        = p3dr.PerspectiveCameras(device=device_torch,
                                  R=camera_rotation,
                                  T=torch.zeros((1, 3), dtype=torch.float32, device=device_torch),
                                  K=K)
    rasterizer = p3dr.MeshRasterizer(cameras, raster_settings=rasterization_settings)
    fragments = rasterizer(mesh_torch)

    print(fragments.pix_to_face[0, 300, 250])  # should print 0
    print(fragments.pix_to_face[0, 200, 350])  # should print 1
    np.save("/home/algomorph/Workbench/NeuralTracking/csrc/tests/test_data/arrays/plane_0_pixel_face_indices.npy",
            fragments.pix_to_face.cpu().numpy().reshape(image_size[0], image_size[1], 1))
    np.save("/home/algomorph/Workbench/NeuralTracking/csrc/tests/test_data/arrays/plane_0_pixel_depths.npy",
            fragments.zbuf.cpu().numpy().reshape(image_size[0], image_size[1], 1))
    np.save(
        "/home/algomorph/Workbench/NeuralTracking/csrc/tests/test_data/arrays/plane_0_pixel_barycentric_coordinates.npy",
        fragments.bary_coords.cpu().numpy().reshape(image_size[0], image_size[1], 3))
    np.save("/home/algomorph/Workbench/NeuralTracking/csrc/tests/test_data/arrays/plane_0_pixel_face_distances.npy",
            fragments.dists.cpu().numpy().reshape(image_size[0], image_size[1], 1))
