#!/usr/bin/python3

# =================================================================================================
# A minimal example that defines a triangle mesh of a cylinder with multiple circular
# cross-sections, riggs it with an armature, and animates the mesh using dual-quaternion skinning.
#
# Copyright 2021 Gregory Kramida
# =================================================================================================

from __future__ import division
import sys
import math
from typing import List

from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d

from functools import reduce
import transformations as tf
from dq3d import quat, dualquat
from dq3d import op

PROGRAM_EXIT_SUCCESS = 0


def compute_anchor_weight_python(anchor_to_point_distance, node_coverage):
    return math.exp(-(anchor_to_point_distance * anchor_to_point_distance) / (2 * node_coverage * node_coverage))


class Viewer:
    def __init__(self):
        cross_section_count = 20
        cylinder_height = 5.0
        self.node_count = 3
        node_coverage = 1.2

        node_y_coordinates = np.linspace(0.0, cylinder_height, self.node_count)
        self.edge_lengths = [node_y_coordinates[i_node + 1] - node_y_coordinates[i_node] for i_node in range(self.node_count - 1)]
        self.node_locations = np.array([[0.0, y, 0.0] for y in node_y_coordinates])

        self.node_dual_quaternions = [dualquat(quat.identity())]
        for edge_length in self.edge_lengths:
            self.node_dual_quaternions.append(dualquat(quat.identity(), [0.0, edge_length, 0.0]))

        self.cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(0.5, 5.0, 20, cross_section_count)
        self.cylinder_mesh.compute_vertex_normals()
        initial_rotation: Rotation = Rotation.from_euler('x', 90, degrees=True)

        self.cylinder_mesh.rotate(initial_rotation.as_matrix())
        self.cylinder_mesh.translate(np.array([0.0, 2.5, 0.0]))
        vertices = np.array(self.cylinder_mesh.vertices)

        self.mesh_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[-2, 0, 0])

        vertex_to_anchor_vectors = np.dstack([vertices - node_location for node_location in self.node_locations])
        vertex_to_anchor_square_distances = np.sum(vertex_to_anchor_vectors ** 2, axis=1)
        self.weights = np.exp(-(vertex_to_anchor_square_distances / (2 * node_coverage * node_coverage)))
        self.vertices = vertices

        self.i_frame = 0
        self.angle_increment = 0.5

    def run_visualizer(self):
        def deform(visualizer):
            self.i_frame += 1
            if self.i_frame > 90 / self.angle_increment:
                return
            per_node_increment = self.angle_increment / (self.node_count - 1)
            node_angles_degrees = [self.i_frame * per_node_increment * i_node for i_node in range(1, self.node_count)]

            # transformations from frame 0 to this frame
            node_rotations = [dualquat(quat(*tf.quaternion_from_euler(0.0, 0.0, np.deg2rad(angle)))) for angle in node_angles_degrees]

            transformed_nodes_dual_quaternions: List[dualquat] = [self.node_dual_quaternions[0]]
            for i_node in range(1, self.node_count):
                original_dq = self.node_dual_quaternions[i_node]
                transformed_dq: dualquat = original_dq * node_rotations[i_node - 1] * original_dq.inverse()
                transformed_nodes_dual_quaternions.append(transformed_dq)

            transformed_vertices = np.array([op.dlb(weights, transformed_nodes_dual_quaternions).transform_point(vertex)
                                             for weights, vertex in zip(self.weights, self.vertices)])
            self.cylinder_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            self.cylinder_mesh.compute_vertex_normals()
            visualizer.update_geometry(self.cylinder_mesh)

            return False

        o3d.visualization.draw_geometries_with_animation_callback([self.cylinder_mesh, self.mesh_coordinate_frame],
                                                                  callback_function=deform,
                                                                  window_name="DQB Skinning 3D test")


def main():
    viewer = Viewer()
    viewer.run_visualizer()
    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
