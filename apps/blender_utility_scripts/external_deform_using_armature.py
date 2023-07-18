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
from apps.geometry_utility_scripts.triangle_indices_to_vertex_edge_array import triangle_indices_to_vertex_edge_array
from enum import Enum

PROGRAM_EXIT_SUCCESS = 0


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


# input section
NODES_SOURCE_OBJECT_NAME = "Plane Nodes Source"
NODES_TARGET_OBJECT_NAME = "Plane Nodes Target"
SKIN_SOURCE_OBJECT_NAME = "Plane Skin Source"
SKIN_TARGET_OBJECT_NAME = "Plane Skin Target"


def deform_objects(
        rotations: List[Rotation] | None = None, node_coverage: float = 0.1,
        rotate_using_normals: bool = False, export_edges: bool = False, fixed_node_coverage: bool = True
):
    # calculation
    objects_to_look_at = None
    if bpy.context.scene.collection.objects.get(NODES_SOURCE_OBJECT_NAME) is not None:
        objects_to_look_at = bpy.context.scene.collection.objects
    else:
        objects_to_look_at = bpy.context.view_layer.active_layer_collection.collection.objects

    nodes_source_mesh = objects_to_look_at.get(NODES_SOURCE_OBJECT_NAME).data
    nodes_target_mesh = objects_to_look_at.get(NODES_TARGET_OBJECT_NAME).data
    skin_source_object = objects_to_look_at.get(SKIN_SOURCE_OBJECT_NAME)

    nodes_source_np = np.array([np.array(v.co) for v in nodes_source_mesh.vertices]).astype(np.float32)

    nodes_source = o3c.Tensor(nodes_source_np)
    nodes_target = o3c.Tensor(np.array([np.array(v.co) for v in nodes_target_mesh.vertices]).astype(np.float32))
    node_translations = nodes_target - nodes_source
    node_count = len(nodes_source)

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

    if fixed_node_coverage or node_count == 1:
        anchors, weights = nnrt.geometry.functional.compute_anchors_and_weights_euclidean_fixed_node_weight(
            skin_source_vertices, nodes_source, 4, 0, node_coverage
        )
    else:
        tree = nnrt.core.KDTree(nodes_source)
        closest, distances = tree.find_k_nearest_to_points(nodes_source, 2, True)
        node_weights = distances[:, 1]
        node_weights *= node_weights
        anchors, weights = nnrt.geometry.functional.compute_anchors_and_weights_euclidean_variable_node_weight(
            skin_source_vertices, nodes_source, node_weights, 4, 0
        )

    skin_source_point_cloud = o3d.t.geometry.PointCloud(skin_source_vertices)
    skin_target_point_cloud = nnrt.geometry.functional.warp_point_cloud(
        skin_source_point_cloud, nodes_source, node_rotations, node_translations, anchors, weights, 0
    )

    skin_target_object = skin_source_object.copy()
    skin_target_object.name = SKIN_TARGET_OBJECT_NAME
    skin_target_object.data = skin_target_object.data.copy()
    skin_target_object.data.name = SKIN_TARGET_OBJECT_NAME
    for vertex, warped_coord in zip(skin_target_object.data.vertices,
                                    skin_target_point_cloud.point["positions"].numpy()):
        vertex.co[0] = warped_coord[0]
        vertex.co[1] = warped_coord[1]
        vertex.co[2] = warped_coord[2]
    objects_to_look_at.link(skin_target_object)


class Scene(Enum):
    SINGLE_NODE_ROTATION = 0
    TWO_NODE_SEPARATE_PLANES_45 = 1
    TWO_NODE_SEPARATE_PLANES_ROTATION_ONLY_5 = 2
    TWO_NODE_SEPARATE_PLANES_MIN_ROTATION_ONLY_5 = 3
    TWO_NODE_COMBINED_PLANES = 4


SceneDescriptor = namedtuple("SceneDescriptor", "starter_file, rotations, node_coverage, output_prefix, fixed_coverage")

generated_blender_test_data_path = "/mnt/Data/Reconstruction/synthetic_data/depth_fitter_tests/"

scene_data_map = {
    Scene.SINGLE_NODE_ROTATION: SceneDescriptor(
        starter_file="fitter_test_starter_file",
        rotations=[Rotation.from_euler('xyz', (-45, 0, 0), degrees=True)],
        node_coverage=0.25,
        output_prefix="plane_fit_1_node_rotation_-45",
        fixed_coverage=True
    ),
    Scene.TWO_NODE_SEPARATE_PLANES_45: SceneDescriptor(
        starter_file="fitter_test_2-node_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (45, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (-45, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="plane_fit_2_nodes_45",
        fixed_coverage=True
    ),
    Scene.TWO_NODE_SEPARATE_PLANES_ROTATION_ONLY_5: SceneDescriptor(
        starter_file="fitter_test_2-node_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (5, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (-5, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="plane_fit_2_nodes_rotation_only_5",
        fixed_coverage=True
    ),
    Scene.TWO_NODE_SEPARATE_PLANES_MIN_ROTATION_ONLY_5: SceneDescriptor(
        starter_file="fitter_test_2-node_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (5, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (-5, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="plane_fit_2_nodes_min_rotation_only_5",
        fixed_coverage=False
    ),
    Scene.TWO_NODE_COMBINED_PLANES: SceneDescriptor(
        starter_file="fitter_test_2-node_connected_starter_file",
        rotations=[
            Rotation.from_euler('xyz', (0, 0, 0), degrees=True),
            Rotation.from_euler('xyz', (0, 0, 0), degrees=True)
        ],
        node_coverage=0.1,
        output_prefix="contiguous_surface_fit_2_nodes",
        fixed_coverage=True
    )
}


def main():
    scene = Scene.TWO_NODE_SEPARATE_PLANES_MIN_ROTATION_ONLY_5
    scene_data = scene_data_map[scene]
    bpy.ops.wm.open_mainfile(
        filepath=f"{generated_blender_test_data_path}{scene_data.starter_file}.blend"
    )

    deform_objects(scene_data.rotations, scene_data.node_coverage, fixed_node_coverage=scene_data.fixed_coverage)

    bpy.ops.wm.save_mainfile(
        filepath=f"{generated_blender_test_data_path}{scene_data.output_prefix}.blend"
    )

    source_object = bpy.context.scene.objects[SKIN_SOURCE_OBJECT_NAME]
    target_object = bpy.context.scene.objects[SKIN_TARGET_OBJECT_NAME]

    source_object.select_set(True)

    bpy.ops.export_mesh.ply(
        filepath=f"{generated_blender_test_data_path}{scene_data.output_prefix}_source.ply",
        use_selection=True,
        use_normals=True,
        use_mesh_modifiers=True,
        use_uv_coords=True,
        axis_forward='Y',
        axis_up='Z',
        use_ascii=False
    )

    source_object.select_set(False)
    target_object.select_set(True)

    bpy.ops.export_mesh.ply(
        filepath=f"{generated_blender_test_data_path}{scene_data.output_prefix}_target.ply",
        use_selection=True,
        use_normals=True,
        use_mesh_modifiers=True,
        use_uv_coords=True,
        axis_forward='Y',
        axis_up='Z',
        use_ascii=False
    )

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
