#!/usr/bin/python3

# =================================================================================================
# A minimal example that defines a triangle mesh of a cylinder with multiple circular
# cross-sections, riggs it with an armature, and animates the mesh using dual-quaternion skinning.
#
# Copyright 2021 Gregory Kramida
# =================================================================================================

from __future__ import division
import sys

from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d

from functools import reduce
import transformations as tf
from dq3d import quat, dualquat
from dq3d import op

PROGRAM_EXIT_SUCCESS = 0


def main():

    mesh = o3d.geometry.TriangleMesh.create_cylinder(0.5, 5.0, 20, 20)
    mesh.compute_vertex_normals()
    rotation: Rotation = Rotation.from_euler('x', 90, degrees=True)
    mesh.rotate(rotation.as_matrix())
    o3d.visualization.draw_geometries([mesh],
                                      front=[0, 0, 1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, 1.0, 0],
                                      zoom=0.7)
    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
