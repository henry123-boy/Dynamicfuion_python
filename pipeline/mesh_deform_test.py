# =================================================================================================
# A minimal example that loads a 3d mesh previously generated using a depth map and image
# (via TSDF integration and marching cubes), deforms it using node transformations
# (to a certain target frame) previously output from NeuralTracking, and compares it to the mesh
# generated from the target frame.
#
# Copyright 2021 Gregory Kramida
#
# Please only run script from repository root
#
# =================================================================================================

import sys
import os

import open3d as o3d
import numpy as np
import options
from model.dataset import StaticCenterCrop, DeformDataset
import transformations as tf
from dq3d import quat, dualquat
from dq3d import op

import nnrt

PROGRAM_EXIT_SUCCESS = 0


def main():
    # Options
    do_dl3d_sanity_check_on_nodes = False

    # Load generated isosurfaces
    mesh200: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000200_red_shorts.ply")
    mesh400: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh("output/mesh_000400_red_shorts.ply")

    # Load graph data
    sequence_directory = "/mnt/Data/Reconstruction/real_data/deepdeform/val/seq014"
    graph_filename = "5db1b1dcfce4e1021deb83dc_shorts_000200_000400_geodesic_0.05"
    image_size = (480, 640)
    cropper = StaticCenterCrop(image_size, (options.image_height, options.image_width))
    graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = \
        DeformDataset.load_graph_data(sequence_directory, graph_filename, False, cropper)

    # Compute vertex anchors & weights
    vertices = np.array(mesh200.vertices)
    vertex_anchors, vertex_weights = nnrt.compute_vertex_anchors_euclidean(graph_nodes, vertices, options.node_coverage)

    # Load graph transformation
    with open("output/red_shorts_shorts_000200_000400_rotations.np", 'rb') as file:
        rotations = np.load(file)

    with open("output/red_shorts_shorts_000200_000400_translations.np", 'rb') as file:
        translations = np.load(file)

    node_transformations_dual_quaternions = np.array([dualquat(quat(rotation), translation) for rotation, translation in
                                                      zip(rotations, translations)])

    if do_dl3d_sanity_check_on_nodes:
        i_node = 0
        for node in graph_nodes:
            print("matrices: ", node, "-->", rotations[i_node].dot(node) + translations[i_node])
            weight_one_hot_vector = np.zeros(len(graph_nodes))
            weight_one_hot_vector[i_node] = 1.0
            print("dual quaternions: ", node, "-->", op.dlb(weight_one_hot_vector, node_transformations_dual_quaternions).transform_point(node))
            i_node += 1

    i_vertex = 0
    deformed_vertices = np.zeros_like(vertices)
    for vertex in vertices:
        vertex_anchor_quaternions = node_transformations_dual_quaternions[vertex_anchors[i_vertex]]
        vertex_anchor_weights = vertex_weights[i_vertex]
        deformed_vertices[i_vertex] = op.dlb(vertex_anchor_weights, vertex_anchor_quaternions).transform_point(vertex)
        i_vertex += 1

    mesh200_transformed = o3d.geometry.TriangleMesh(o3d.cuda.pybind.utility.Vector3dVector(deformed_vertices), mesh200.triangles)
    mesh200_transformed.compute_vertex_normals()

    # o3d.visualization.draw_geometries([mesh200],
    #                                   front=[0, 0, -1],
    #                                   lookat=[0, 0, 1.5],
    #                                   up=[0, -1.0, 0],
    #                                   zoom=0.7)

    # TODO: add visualization toggle switch between meshes, use shortcuts: T(ransformed), G(round truth),B(oth)
    o3d.visualization.draw_geometries([mesh400, mesh200_transformed],
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)

    # TODO: Compare deformed & target mesh numerically

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
