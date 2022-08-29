#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 8/24/22.
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
from typing import Union

import torch
import torch.nn.functional as tnn_func
import pytorch3d.structures as p3ds


def warp_meshes_using_node_anchors(canonical_meshes: p3ds.Meshes,
                                   graph_nodes: torch.Tensor,
                                   graph_node_rotations: torch.Tensor,
                                   graph_node_translations: torch.Tensor,
                                   mesh_vertex_anchors: torch.Tensor,
                                   mesh_vertex_anchor_weights: torch.Tensor,
                                   extrinsic_matrix: Union[torch.Tensor, None] = None) -> p3ds.Meshes:
    anchor_count = mesh_vertex_anchors.shape[1]
    # (vertex_count, 3)
    mesh_vertices: torch.Tensor = canonical_meshes.verts_packed()
    if extrinsic_matrix is not None:
        camera_rotation = extrinsic_matrix[0:3, 0:3]
        camera_translation = extrinsic_matrix[0:3, 3]
        mesh_vertices = mesh_vertices.mm(camera_rotation.T)
        mesh_vertices += camera_translation.T

    vertex_count = mesh_vertices.shape[0]
    # (vertex_count, anchor_count, 3)
    tiled_vertices = mesh_vertices.view(mesh_vertices.shape[0], 1,
                                        mesh_vertices.shape[1]).tile(1, anchor_count, 1)
    mesh_vertex_anchors_long = mesh_vertex_anchors.to(torch.long)
    # (vertex_count, anchor_count, 3)
    sampled_nodes = graph_nodes[mesh_vertex_anchors_long]
    # (vertex_count, anchor_count, 3, 3)
    sampled_rotations = graph_node_rotations[mesh_vertex_anchors_long]
    # (vertex_count, anchor_count, 3)
    sampled_translations = graph_node_translations[mesh_vertex_anchors_long]

    # (vertex_count, 3)
    warped_vertices = (
            mesh_vertex_anchor_weights.view(vertex_count, anchor_count, 1) *
            (
                    sampled_nodes +
                    torch.matmul(sampled_rotations,
                                 (tiled_vertices - sampled_nodes).view(vertex_count, anchor_count, 3, 1))
                    .view(vertex_count, anchor_count, 3)
                    + sampled_translations
            )
    ).sum(dim=1)

    # (vertex_count, 3)
    mesh_vertex_normals = canonical_meshes.verts_normals_packed()
    # (vertex_count, anchor_count, 3)
    tiled_normals = mesh_vertex_normals.view(mesh_vertex_normals.shape[0], 1,
                                             mesh_vertex_normals.shape[1]).tile(1, anchor_count, 1)
    # (vertex_count, 3)
    warped_normals = tnn_func.normalize(
        (
                mesh_vertex_anchor_weights.view(vertex_count, anchor_count, 1) *
                (
                        sampled_nodes +
                        torch.matmul(sampled_rotations, tiled_normals.view(vertex_count, anchor_count, 3, 1))
                        .view(vertex_count, anchor_count, 3)
                )
        ).sum(dim=1), dim=1
    )
    return p3ds.Meshes(warped_vertices.unsqueeze(0),
                       canonical_meshes.faces_padded(),
                       textures=canonical_meshes.textures,
                       verts_normals=warped_normals.unsqueeze(0))
