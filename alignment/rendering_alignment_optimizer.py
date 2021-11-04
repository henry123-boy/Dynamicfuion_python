#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/29/21.
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

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

from warp_field.graph import DeformationGraphOpen3D
from nnrt.geometry import WarpableTSDFVoxelGrid
from linear_solver_lu import LinearSolverLU
from settings import GraphParameters
from rendering.pytorch3d_renderer import PyTorch3DRenderer, RenderMaskCode


class RenderingAlignmentOptimizer:
    def __init__(self, image_size_hw: Tuple[int, int], device: o3c.Device, intrinsic_matrix: o3c.Tensor):
        self.renderer = PyTorch3DRenderer(image_size_hw, device, intrinsic_matrix)
        pass

    def termination_condition_reached(self) -> bool:
        raise NotImplementedError("Not implemented")

    def optimize_graph(self, graph: DeformationGraphOpen3D, tsdf: WarpableTSDFVoxelGrid,
                       target_points: o3d.t.geometry.PointCloud, target_rgb: o3d.t.geometry.Image):
        canonical_mesh: o3d.t.geometry.TriangleMesh = tsdf.extract_surface_mesh(-1, 0)
        while not self.termination_condition_reached():
            warped_mesh = graph.warp_mesh_mat(canonical_mesh, GraphParameters.node_coverage.value)

            _, rendered_color = self.renderer.render_mesh(warped_mesh, render_mask=RenderMaskCode.RGB)

            # point-to-plane error



