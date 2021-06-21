# ==================================================================================================
# A toy code example that tests forward-warping voxel centers using dual quaternion blending & an existing
# sparse motion graph
#
# Please run script from repository root, i.e.:
# python3 ./pipeline/warp_voxel_centers_dq3d_test.py
#
# Copyright 2021 Gregory Kramida
# ==================================================================================================

import open3d as o3d
import sys
import os
import numpy as np
from dq3d import quat, dualquat

import data.io as io
import options
import graph
from data import StandaloneFramePreset, StandaloneFrameDataset, camera

from pipeline.numba_cuda.fusion_functions import cuda_compute_voxel_center_anchors, cuda_compute_psdf_warped_voxel_centers
from utils.hardware_id import get_mac_address

PROGRAM_EXIT_SUCCESS = 0


def main():
    with open(os.path.join(options.output_directory, "voxel_centers_000200_red_shorts.np"), 'rb') as file:
        voxel_centers = np.load(file)

    # TODO: is the original pre-generated graph radically different due to some camera frustum offset during generation?
    # graph_nodes = utils.load_graph_nodes(
    #     os.path.join(options.dataset_base_dir,
    #                  "val/seq014/graph_nodes/5db1b1dcfce4e1021deb83dc_shorts_000200_000400_geodesic_0.05.bin"))

    mesh200: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000200_red_shorts.ply")
    graph200 = graph.build_deformation_graph_from_mesh(mesh200, node_coverage=options.node_coverage)
    graph_nodes = graph200.nodes

    # Load graph transformation
    with open(os.path.join(options.output_directory, "red_shorts_shorts_000200_000400_rotations.np"), 'rb') as file:
        rotations = np.load(file)

    with open(os.path.join(options.output_directory, "red_shorts_shorts_000200_000400_translations.np"), 'rb') as file:
        translations = np.load(file)

    node_transformations_dual_quaternions = [dualquat(quat(rotation), translation) for rotation, translation in
                                             zip(rotations, translations)]
    node_transformations_dual_quaternions = np.array([np.concatenate((dq.real.data, dq.dual.data)) for dq in
                                                      node_transformations_dual_quaternions])

    camera_rotation = np.ascontiguousarray(np.eye(3, dtype=np.float32))
    camera_translation = np.ascontiguousarray(np.zeros(3, dtype=np.float32))

    voxel_center_anchors, voxel_center_weights = \
        cuda_compute_voxel_center_anchors(voxel_centers, graph_nodes, camera_rotation, camera_translation, options.node_coverage)

    red_shorts_400_frame: StandaloneFrameDataset = StandaloneFramePreset.RED_SHORTS_400.value

    depth_image = np.array(o3d.io.read_image(red_shorts_400_frame.get_depth_image_path()))
    intrinsic_matrix = np.loadtxt(red_shorts_400_frame.get_intrinsics_path())

    voxel_psdf = cuda_compute_psdf_warped_voxel_centers(depth_image, intrinsic_matrix, camera_rotation, camera_translation,
                                                        voxel_centers, voxel_center_anchors, voxel_center_weights,
                                                        node_transformations_dual_quaternions)

    voxel_psdf_non_nan = voxel_psdf[np.logical_not(np.isnan(voxel_psdf))]
    np.set_printoptions(suppress=True)
    print(voxel_psdf_non_nan[:100])

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
