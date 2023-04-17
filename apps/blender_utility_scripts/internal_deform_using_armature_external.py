#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 2/9/23.
#  Copyright (c) 2023 Gregory Kramida
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

# script intended for usage outside of blender

import sys
from collections import namedtuple
from typing import List

import bpy
import numpy as np
import nnrt
import open3d as o3d
import open3d.core as o3c
from scipy.spatial.transform import Rotation
from triangle_indices_to_vertex_edge_array import triangle_indices_to_vertex_edge_array
from enum import Enum

PROGRAM_EXIT_SUCCESS = 0


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def deform_objects(
        rotations: List[Rotation] | None = None, node_coverage: float = 0.1,
        rotate_using_normals: bool = False, export_edges: bool = False
):
    # input section
    nodes_source_object_name = "Plane Nodes Source"
    nodes_target_object_name = "Plane Nodes Target"
    skin_source_object_name = "Plane Skin Source"
    node_coverage = 0.1

    # calculation
    objects_to_look_at = None
    if bpy.context.scene.collection.objects.get(nodes_source_object_name) is not None:
        objects_to_look_at = bpy.context.scene.collection.objects
    else:
        objects_to_look_at = bpy.context.view_layer.active_layer_collection.collection.objects

    nodes_source_mesh = objects_to_look_at.get(nodes_source_object_name).data
    nodes_target_mesh = objects_to_look_at.get(nodes_target_object_name).data
    skin_source_object = objects_to_look_at.get(skin_source_object_name)

    nodes_source_np = np.array([np.array(v.co) for v in nodes_source_mesh.vertices]).astype(np.float32)

    nodes_source = o3c.Tensor(nodes_source_np)
    nodes_target = o3c.Tensor(np.array([np.array(v.co) for v in nodes_target_mesh.vertices]).astype(np.float32))
    node_translations = nodes_target - nodes_source

    if len(nodes_source) != len(nodes_target):
        raise ValueError(
            f"Number of vertices in node source ({len(nodes_source)}) must match the number of vertices in node target {len(nodes_target)}")

    # for when edge export is necessary
    if export_edges:
        edges_np = triangle_indices_to_vertex_edge_array(
            np.array([np.array(e.vertices) for e in nodes_source_mesh.edges]))
        print(repr(edges_np))

    if rotate_using_normals:
        normals_source_np = np.array([np.array(v.normal) for v in nodes_source_mesh.vertices]).astype(np.float32)
        normals_target_np = np.array([np.array(v.normal) for v in nodes_target_mesh.vertices]).astype(np.float32)
        V = np.cross(normals_source_np, normals_target_np)
        S = np.linalg.norm(V, axis=1)
        C = np.einsum('ij, ij->i', normals_source_np, normals_target_np)
        node_rotations_np = np.array(
            [np.eye(3) + skew(v) + skew(v).dot(skew(v)) * (1 - c) / s ** 2 for v, s, c in zip(V, S, C)])
        node_rotations = o3c.Tensor(node_rotations_np.astype(np.float32))
    else:
        if rotations is not None:
            if len(rotations) != len(nodes_source):
                raise ValueError(
                    f"Number of rotations ({len(rotations)}) must match the number of vertices in node source {len(nodes_source)}")
            node_rotations = o3c.Tensor(
                np.array([rotation.as_matrix().astype(np.float32) for rotation in rotations]))
        else:
            node_rotations = o3c.Tensor(np.array([np.eye(3, dtype=np.float32)] * len(nodes_source)))

    skin_source_vertices = o3c.Tensor(
        np.array([np.array(v.co) for v in skin_source_object.data.vertices]).astype(np.float32)
    )

    anchors, weights = nnrt.geometry.functional.compute_anchors_and_weights_euclidean(
        skin_source_vertices, nodes_source, 4, 0, node_coverage
    )
    skin_target_object_name = "Plane Skin Target"
    skin_source_point_cloud = o3d.t.geometry.PointCloud(skin_source_vertices)
    skin_target_point_cloud = nnrt.geometry.functional.warp_point_cloud(
        skin_source_point_cloud, nodes_source, node_rotations, node_translations, anchors, weights, 0
    )

    skin_target_object = skin_source_object.copy()
    skin_target_object.name = skin_target_object_name
    skin_target_object.data = skin_target_object.data.copy()
    skin_target_object.data.name = skin_target_object_name
    for vertex, warped_coord in zip(skin_target_object.data.vertices,
                                    skin_target_point_cloud.point["positions"].numpy()):
        vertex.co[0] = warped_coord[0]
        vertex.co[1] = warped_coord[1]
        vertex.co[2] = warped_coord[2]
    objects_to_look_at.link(skin_target_object)


class Scene(Enum):
    SINGLE_NODE_ROTATION = 0
    TWO_NODE_SEPARATE_PLANES = 1
    TWO_NODE_COMBINED_PLANES = 2


SceneDescriptor = namedtuple("SceneDescriptor", "starter_file, rotations, node_coverage, output_prefix")

generated_blender_test_data_path = "/mnt/Data/Reconstruction/synthetic_data/depth_fitter_tests/"

scene_data_map = {
    Scene.SINGLE_NODE_ROTATION: SceneDescriptor(
        starter_file="fitter_test_starter_file",
        rotations=[Rotation.from_euler('xyz', (-45, 0, 0), degrees=True)],
        node_coverage=0.25,
        output_prefix="plane_fit_1_node_rotation_-45"
    ),
    Scene.TWO_NODE_SEPARATE_PLANES: SceneDescriptor(
        starter_file="fitter_test_2-node_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (45, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (-45, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="plane_fit_2_nodes_45"
    ),
    Scene.TWO_NODE_COMBINED_PLANES: SceneDescriptor(
        starter_file="fitter_test_2-node_connected_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (0, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (0, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="contiguous_surface_fit_2_nodes"
    )
}


def main():
    scene = Scene.TWO_NODE_COMBINED_PLANES
    scene_data = scene_data_map[scene]
    bpy.ops.wm.open_mainfile(
        filepath=f"{generated_blender_test_data_path}{scene_data.starter_file}.blend"
    )

    deform_objects(scene_data.rotations, scene_data.node_coverage)

    bpy.ops.wm.save_mainfile(
        filepath=f"{generated_blender_test_data_path}{scene_data.output_prefix}.blend"
    )

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
