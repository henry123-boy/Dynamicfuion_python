#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/29/21.
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import torch
import open3d as o3d
import open3d.core as o3c
from typing import Tuple
import torch.utils.dlpack as torch_dlpack
import sys


from warp_field.graph_warp_field import GraphWarpField
from nnrt.geometry import NonRigidSurfaceVoxelBlockGrid, compute_point_to_plane_distances
from alignment.common import LinearSolverLU
from settings import GraphParameters
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode
from alignment.render_based.loss_penalty_functions import apply_data_residual_penalty






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
