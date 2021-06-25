import math

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

    # @formatter:off
    nodes = np.array([
        [-9.9, -1.2, 0],  # 0
        [-4.3,  1.3, 0],  # 1
        [-3.7,  0.5, 0],  # 2
        [-3.6, -0.5, 0],  # 3
        [-3.8, -1.4, 0],  # 4
        [ 1.0,  0.4, 0],  # 5
        [ 3.3,  0.6, 0],  # 6
        [-8.4,  5.5, 0],  # 7
        [-6.6,  6.5, 0],  # 8
        [-5.3,  5.3, 0],  # 9
        [-11.5, 4.9, 0],  # 10
    ], dtype=np.float32)

    edges = np.array([
        [1,  4, -1, -1],  # 0
        [0,  2, -1, -1],  # 1
        [1,  3, -1, -1],  # 2
        [2,  4, -1, -1],  # 3
        [0,  3, -1, -1],  # 4
        [6, -1, -1, -1],  # 5
        [5, -1, -1, -1],  # 6
        [8, 10, -1, -1],  # 7
        [7,  9, -1, -1],  # 8
        [8, -1, -1, -1],  # 9
        [7, -1, -1, -1],  # 10
    ], dtype=np.int32)

    dist_0_1 =  np.linalg.norm(nodes[0] - nodes[1])
    dist_0_4 =  np.linalg.norm(nodes[0] - nodes[4])
    dist_1_2 =  np.linalg.norm(nodes[1] - nodes[2])
    dist_2_3 =  np.linalg.norm(nodes[2] - nodes[3])
    dist_3_4 =  np.linalg.norm(nodes[3] - nodes[4])
    dist_5_6 =  np.linalg.norm(nodes[5] - nodes[6])
    dist_7_8 =  np.linalg.norm(nodes[7] - nodes[8])
    dist_7_10 = np.linalg.norm(nodes[7] - nodes[10])
    dist_8_9 =  np.linalg.norm(nodes[8] - nodes[9])

    dist_s0_4 = np.linalg.norm(vertices[0] - nodes[4])
    dist_s0_5 = np.linalg.norm(vertices[0] - nodes[5])
    dist_s0_9 = np.linalg.norm(vertices[0] - nodes[9])

    anchors_gt = np.array([[0, 1, 2, 3, 4, 5, 6, 9],
                           [1, 2, 3, 4, 7, 8, 9, 10]], dtype=np.int32)

    sp_dists_s0 = np.array([
        dist_s0_4 + dist_0_4,                        # anchor node: 0
        dist_s0_4 + dist_3_4 + dist_2_3 + dist_1_2,  # anchor node: 1
        dist_s0_4 + dist_3_4 + dist_2_3,             # anchor node: 2
        dist_s0_4 + dist_3_4,                        # anchor node: 3
        dist_s0_4,                                   # anchor node: 4
        dist_s0_5,                                   # anchor node: 5
        dist_s0_5 + dist_5_6,                        # anchor node: 6
        dist_s0_9,                                   # anchor node: 9
    ])
    # @formatter:on

    def compute_anchor_weight(dist, _node_coverage):
        return math.exp(-(math.pow(dist, 2.0) / 2.0 * math.pow(_node_coverage, 2.0)))

    u_compute_anchor_weights = np.vectorize(compute_anchor_weight)
    weights_s0_gt = u_compute_anchor_weights(sp_dists_s0, node_coverage)
    weight_sum = weights_s0_gt.sum()
    weights_s0_gt /= weight_sum

    anchors, weights = nnrt.compute_vertex_anchors_shortest_path(vertices, nodes, edges, anchor_count, node_coverage)

    assert np.alltrue(anchors_gt == anchors)
    assert np.allclose(weights.sum(axis=1), [[1.0], [1.0]])
    assert np.allclose(weights[0], weights_s0_gt)
