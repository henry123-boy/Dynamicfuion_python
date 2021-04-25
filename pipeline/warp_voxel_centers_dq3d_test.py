# ==================================================================================================
# A toy code example that tests forward-warping voxel centers using dual quaternion blending & an existing
# sparse motion graph
#
# Please run script from repository root, i.e.:
# python3 ./pipeline/warp_voxel_centers_dq3d_test.py
#
# Copyright 2021 Gregory Kramida
# ==================================================================================================

import sys
import os
import numpy as np
from dq3d import quat, dualquat
from dq3d import op
import options

import utils.utils as utils

from pipeline.numba_cuda.compute_voxel_anchors import cuda_compute_voxel_center_anchors

PROGRAM_EXIT_SUCCESS = 0


def main():
    with open(os.path.join(options.experiments_dir, "voxel_centers_000200_red_shorts.np"), 'rb') as file:
        voxel_centers = np.load(file)

    # TODO: these graph nodes are all too far away (due to some voxel volume / camera frustum offset, maybe??
    #  see if you can generate a local graph instead using the TSDF / extraction / sampling technique
    graph_nodes = utils.load_graph_nodes(
        os.path.join(options.dataset_base_dir,
                     "val/seq014/graph_nodes/5db1b1dcfce4e1021deb83dc_shorts_000200_000400_geodesic_0.05.bin"))

    # Load graph transformation
    with open(os.path.join(options.experiments_dir, "red_shorts_shorts_000200_000400_rotations.np"), 'rb') as file:
        rotations = np.load(file)

    with open(os.path.join(options.experiments_dir, "red_shorts_shorts_000200_000400_translations.np"), 'rb') as file:
        translations = np.load(file)

    node_transformations_dual_quaternions = np.array([dualquat(quat(rotation), translation) for rotation, translation in
                                                      zip(rotations, translations)])

    camera_rotation = np.ascontiguousarray(np.eye(3, dtype=np.float32))
    camera_translation = np.ascontiguousarray(np.zeros(3, dtype=np.float32))

    voxel_center_weights, voxel_center_anchors = cuda_compute_voxel_center_anchors(voxel_centers, graph_nodes,
                                                                                   camera_rotation, camera_translation,
                                                                                   options.node_coverage)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
