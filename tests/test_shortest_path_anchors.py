import numpy as np
import open3d as o3d
import nnrt

from pipeline.numpy_cpu.distance_matrix import compute_distance_matrix


# see tests/test_data/AnchorComputationTest1.jpg for reference
def test_shortest_path_anchors1():
    anchor_count = 8
    node_coverage = 1.0
    vertices = np.array([
        [-4.6, -4.2, 0],  # SOURCE_1
        [-4.4, 7.1, 0]  # SOURCE_2
    ], dtype=np.float32)
    nodes = np.array([
        [-9.9, -1.2, 0],  # A 0
        [-4.3, 1.3, 0],  # B 1
        [-3.7, 0.5, 0],  # C 2
        [-3.6, -0.5, 0],  # D 3
        [-3.8, -1.4, 0],  # E 4
        [1.0, 0.4, 0],  # F 5
        [3.3, 0.6, 0],  # G 6
        [-8.4, 5.5, 0],  # H 7
        [-6.6, 6.5, 0],  # I 8
        [-5.3, 5.3, 0],  # J 9
        [-11.5, 4.9, 0],  # K 10
    ], dtype=np.float32)

    edges = np.array([
        [1, 4, -1, -1],  # A
        [0, 2, -1, -1],  # B
        [1, 3, -1, -1],  # C
        [2, 4, -1, -1],  # D
        [0, 3, -1, -1],  # E
        [6, -1, -1, -1],  # F
        [5, -1, -1, -1],  # G
        [8, 10, -1, -1],  # H
        [7, 9, -1, -1],  # I
        [8, -1, -1, -1],  # J
        [7, -1, -1, -1],  # K
    ], dtype=np.int32)
    anchors, weights = nnrt.compute_vertex_anchors_shortest_path(vertices, nodes, edges, anchor_count, node_coverage)

    anchors_gt = np.array([[0, 1, 2, 3, 4, 5, 6, 9],
                           [0, 1, 2, 3, 7, 8, 9, 10]], dtype=np.int32)

    assert np.alltrue(anchors_gt == anchors)
    assert np.allclose(weights.sum(axis=1), [[1.0], [1.0]])
