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

from pytorch3d.structures import Meshes


def warp_meshes_using_node_anchors() -> Meshes:
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