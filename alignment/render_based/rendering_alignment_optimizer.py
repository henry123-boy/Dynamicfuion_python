#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/29/21.
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
import sys

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import open3d as o3d
import open3d.core as o3c
from typing import Tuple
import torch.utils.dlpack as torch_dlpack
import pytorch3d.structures as p3d_struct
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftPhongShader, PointLights, \
    PerspectiveCameras

from warp_field.graph_warp_field import GraphWarpField
from nnrt.geometry import NonRigidSurfaceVoxelBlockGrid, compute_point_to_plane_distances, \
    compute_anchors_and_weights_shortest_path
from alignment.common import LinearSolverLU
from settings import GraphParameters
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode
import rendering.converters


class PureTorchOptimizer:
    def __init__(self, reference_rgb_image: o3d.t.geometry.Image, reference_point_cloud: o3d.t.geometry.PointCloud,
                 intrinsic_matrix: o3c.Tensor, extrinsic_matrix: o3c.Tensor,
                 warp_field: GraphWarpField, canonical_mesh: o3d.t.geometry.TriangleMesh):
        self.anchor_count = 4
        self.node_coverage = 0.05
        self.reference_rgb_image = \
            torch_dlpack.from_dlpack(reference_rgb_image.as_tensor().to_dlpack()).to(torch.float32) / 255.0
        self.reference_points = torch_dlpack.from_dlpack(reference_point_cloud.point["positions"].to_dlpack())
        self.device = self.reference_rgb_image.device

        self.canonical_meshes = rendering.converters.open3d_mesh_to_pytorch3d(canonical_mesh)

        self.intrinsic_matrix = torch_dlpack.from_dlpack(intrinsic_matrix.to_dlpack()).to(self.device)
        self.extrinsic_matrix = torch_dlpack.from_dlpack(extrinsic_matrix.to_dlpack()).to(self.device)

        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]

        self.graph_nodes = torch_dlpack.from_dlpack(warp_field.nodes.to_dlpack())
        self.graph_edges = torch_dlpack.from_dlpack(warp_field.edges.to_dlpack())
        self.graph_node_rotations = torch.from_dlpack(warp_field.get_node_rotations().to_dlpack())
        self.graph_node_translations = torch.from_dlpack(warp_field.get_node_translations().to_dlpack())

        # note: assume weights are already normalized and weight rows add up to 1.0 skipping places where anchors are -1
        anchors, weights = compute_anchors_and_weights_shortest_path(canonical_mesh.vertex["positions"],
                                                                     warp_field.nodes, warp_field.edges,
                                                                     anchor_count=self.anchor_count,
                                                                     node_coverage=self.node_coverage)
        # (vertex_count, anchor_count)
        self.mesh_vertex_anchors = torch_dlpack.from_dlpack(anchors)
        # (vertex_count, anchor_count)
        self.mesh_vertex_anchor_weights = torch_dlpack.from_dlpack(weights)
        self.mesh_vertex_anchor_weights[self.mesh_vertex_anchors == -1] = 0.0
        self.mesh_vertex_anchors[self.mesh_vertex_anchors == -1] = 0

        self.image_size = (reference_rgb_image.rows, reference_rgb_image.columns)
        pixel_y_coordinates = torch.arange(0, self.image_size[0])
        pixel_x_coordinates = torch.arange(0, self.image_size[1])
        self.pixel_xy_coordinates = torch.dstack(
            torch.meshgrid(pixel_x_coordinates, pixel_y_coordinates, indexing='xy')
        ).reshape(-1, 2)
        self.cameras: PerspectiveCameras = \
            rendering.converters.build_pytorch3d_cameras(self.image_size, intrinsic_matrix, extrinsic_matrix,
                                                         self.device)
        # set up renderer
        lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                             specular_color=((0.0, 0.0, 0.0),), device=self.torch_device,
                             location=[[0.0, 0.0, -3.0]])

        rasterization_settings = RasterizationSettings(image_size=self.image_size,
                                                       cull_backfaces=True,
                                                       cull_to_frustum=True,
                                                       z_clip_value=0.5,
                                                       faces_per_pixel=1)

        self.rasterizer = MeshRasterizer(self.cameras, raster_settings=rasterization_settings)

        self.shader = SoftPhongShader(
            device=self.torch_device,
            cameras=self.cameras,
            lights=lights
        )

    def warp_mesh(self):
        # (vertex_count, 3)
        mesh_vertices = self.canonical_meshes.verts_packed()
        # (vertex_count, anchor_count, 3)
        tiled_vertices = mesh_vertices.view(mesh_vertices.shape[0], 1,
                                            mesh_vertices.shape[1]).tile(1, self.anchor_count, 1)
        # (vertex_count, anchor_count, 3)
        sampled_nodes = self.graph_nodes[self.mesh_vertex_anchors]
        # (vertex_count, anchor_count, 3, 3)
        sampled_rotations = self.graph_node_rotations[self.mesh_vertex_anchors]
        # (vertex_count, anchor_count, 3)
        sampled_translations = self.graph_node_translations[self.mesh_vertex_anchors]

        # (vertex_count, 3)
        warped_vertices = (
                self.mesh_vertex_anchor_weights * sampled_nodes +
                torch.matmul(sampled_rotations, (tiled_vertices - sampled_nodes))
                + sampled_translations
        ).sum(2)

        # (vertex_count, 3)
        mesh_vertex_normals = self.canonical_meshes.verts_normals_packed()
        # (vertex_count, anchor_count, 3)
        tiled_normals = mesh_vertex_normals.view(mesh_vertex_normals.shape[0], 1,
                                                 mesh_vertex_normals.shape[1]).tile(1, self.anchor_count, 1)
        # (vertex_count, 3)
        warped_normals = nn_func.normalize((
                                                   self.mesh_vertex_anchor_weights * sampled_nodes +
                                                   torch.matmul(sampled_rotations, tiled_normals)
                                           ).sum(2), dim=2)
        return p3d_struct.Meshes(warped_vertices.unsqueeze(0),
                                 self.canonical_meshes.faces_padded(),
                                 textures=self.canonical_meshes.textures,
                                 verts_normals=warped_normals.unsqueeze(0))

    def point_to_plane_distances(self, rendered_rgb_image: torch.Tensor, rendered_points: torch.Tensor,
                                 rendered_normals: torch.Tensor) -> torch.Tensor:
        return torch.square((rendered_normals * (rendered_points - self.reference_points)).sum(dim=-1))

    def optimize(self):
        max_iteration_count = 5

        for iteration in range(0, max_iteration_count):
            warped_mesh = self.warp_mesh()
            fragments = self.rasterizer(warped_mesh)

            faces = warped_mesh.faces_packed()  # (F, 3)
            vertex_normals = warped_mesh.verts_normals_packed()  # (V, 3)
            face_normals = vertex_normals[faces]
            rendered_normals = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_normals
            )
            # (image_height, image_width, 3)
            rendered_rgb_image = self.shader(fragments, warped_mesh)[0, ..., :3]
            rendered_points = self.cameras.unproject_points(
                torch.cat((self.pixel_xy_coordinates, fragments.zbuf), dim=1)
            )
            reciprocals = self.point_to_plane_distances(rendered_rgb_image, rendered_points, rendered_normals)
            loss = torch.square(reciprocals)
            gradients = torch.autograd.grad(outputs=(loss,),
                                            inputs=(self.graph_node_rotations, self.graph_node_translations))


class RenderingAlignmentOptimizer:
    def __init__(self, image_size_hw: Tuple[int, int], device: o3c.Device, intrinsic_matrix: o3c.Tensor):
        self.renderer = PyTorch3DRenderer(image_size_hw, device, intrinsic_matrix)
        pass

    def termination_condition_reached(self) -> bool:
        raise NotImplementedError("Not implemented")

    def optimize_graph(self, graph: GraphWarpField, tsdf: NonRigidSurfaceVoxelBlockGrid,
                       target_points: o3d.t.geometry.PointCloud, target_rgb: o3d.t.geometry.Image):
        canonical_mesh: o3d.t.geometry.TriangleMesh = tsdf.extract_surface_mesh(-1, 0)
        while not self.termination_condition_reached():
            warped_mesh = graph.warp_mesh(canonical_mesh, GraphParameters.node_coverage.value)

            rendered_depth, rendered_color = \
                self.renderer.render_mesh(warped_mesh, render_mode_mask=RenderMaskCode.RGB | RenderMaskCode.DEPTH)

            rendered_depth_o3d = o3c.Tensor.from_dlpack(torch_dlpack.to_dlpack(rendered_depth))
            rendered_depth_o3d_image = o3d.t.geometry.Image(rendered_depth_o3d)
            rendered_point_cloud = o3d.t.geometry.PointCloud.create_from_depth_image(rendered_depth_o3d, )

            point_to_plane_distances_o3d = compute_point_to_plane_distances(warped_mesh, target_points)
            point_to_plane_distances_torch = torch_dlpack.from_dlpack(point_to_plane_distances_o3d.to_dlpack())
            data_residuals = apply_data_residual_penalty(point_to_plane_distances_torch)

            rotations = torch_dlpack.from_dlpack(graph.get_node_rotations().to_dlpack())

            data_jacobian = torch.zeros((data_residuals.shape[0], len(graph.nodes) * 6), dtype=rotations.dtype,
                                        device=rotations.device)  # (target point count, node count * 6)
            # TODO: compute jacobian

            # TODO: hierarchical graph structures, regularization residuals, etc. (See #22 for full DF task list)

            raise NotImplementedError("Not yet implemented")
