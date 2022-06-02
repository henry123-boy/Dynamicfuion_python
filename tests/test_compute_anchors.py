from pathlib import Path

import math
import open3d as o3d
import open3d.core as o3c
import nnrt
import pytest
import numpy as np

from data import StaticCenterCrop, DeformDataset, camera
import image_processing


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_compute_anchors_euclidean_nnrtf_vs_legacy_nnrt_cpp(device):
    mesh = o3d.geometry.TriangleMesh.create_box()
    # mesh = mesh.subdivide_midpoint(number_of_iterations=1)
    nodes = np.array([[0., 0., 0.],  # 0
                      [1., 0., 0.],  # 1
                      [0., 0., 1.],  # 2
                      [1., 0., 1.],  # 3
                      [0., 1., 0.],  # 4
                      [1., 1., 0.],  # 5
                      [0., 1., 1.],  # 6
                      [1., 1., 1.]],  # 7
                     dtype=np.float32)
    nodes += np.array([0.1, 0.2, 0.3])  # to make distances unique
    nodes_o3d = o3c.Tensor(nodes, device=device)

    points = np.array(mesh.vertices, dtype=np.float32)
    points_o3d = o3c.Tensor(points, device=device)

    anchor_count = 4

    nodes_tiled = np.tile(nodes, (len(points), 1, 1))
    points_tiled = np.tile(points, (1, len(nodes))).reshape(len(points), -1, 3)
    distance_matrix = np.linalg.norm(points_tiled - nodes_tiled, axis=2)
    distance_sorted_node_indices = distance_matrix.argsort(axis=1)  # distances of closest nodes to each point
    vertex_anchors_gt = np.sort(distance_sorted_node_indices[:, :anchor_count], axis=1)
    distances_sorted = np.take_along_axis(distance_matrix, distance_sorted_node_indices, axis=1)
    distances_gt = distances_sorted[:, :anchor_count]

    old_vertex_anchors, old_vertex_weights = nnrt.compute_vertex_anchors_euclidean(nodes, points, 0.5)
    old_vertex_anchors_sorted = np.sort(old_vertex_anchors, axis=1)
    old_vertex_weights_sorted = np.sort(old_vertex_weights, axis=1)
    vertex_anchors, vertex_weights = nnrt.geometry.compute_anchors_and_weights_euclidean(points_o3d, nodes_o3d,
                                                                                         anchor_count, 0, 0.5)
    vertex_anchors_sorted = np.sort(vertex_anchors.cpu().numpy(), axis=1)
    vertex_weights_sorted = np.sort(vertex_weights.cpu().numpy(), axis=1)

    assert np.allclose(vertex_anchors_gt, old_vertex_anchors_sorted)
    assert np.allclose(vertex_anchors_sorted, old_vertex_anchors_sorted)
    assert np.allclose(vertex_weights_sorted, old_vertex_weights_sorted)


# see anchor_assignment_and_weighing/AnchorComputationTestManual_Reference.jpg for node diagram
def prepare_test_data_anchor_computation_manual(node_coverage: float):
    vertices = np.array([
        [-4.6, -4.2, 0],  # SOURCE_1 in image, s0 here
        [-4.4, 7.1, 0]  # SOURCE_2 in image, s1 here
    ], dtype=np.float32)

    # @formatter:off
    nodes = np.array([
        [-9.9, -1.2, 0],  # 0
        [-4.3, 1.3, 0],  # 1
        [-3.7, 0.5, 0],  # 2
        [-3.6, -0.5, 0],  # 3
        [-3.8, -1.4, 0],  # 4
        [1.0, 0.4, 0],  # 5
        [3.3, 0.6, 0],  # 6
        [-8.4, 5.5, 0],  # 7
        [-6.6, 6.5, 0],  # 8
        [-5.3, 5.3, 0],  # 9
        [-11.5, 4.9, 0],  # 10
    ], dtype=np.float32)

    edges = np.array([
        [1, 4, -1, -1],  # 0
        [0, 2, -1, -1],  # 1
        [1, 3, -1, -1],  # 2
        [2, 4, -1, -1],  # 3
        [0, 3, -1, -1],  # 4
        [6, -1, -1, -1],  # 5
        [5, -1, -1, -1],  # 6
        [8, 10, -1, -1],  # 7
        [7, 9, -1, -1],  # 8
        [8, -1, -1, -1],  # 9
        [7, -1, -1, -1],  # 10
    ], dtype=np.int32)

    dist_0_1 = np.linalg.norm(nodes[0] - nodes[1])
    dist_0_4 = np.linalg.norm(nodes[0] - nodes[4])
    dist_1_2 = np.linalg.norm(nodes[1] - nodes[2])
    dist_2_3 = np.linalg.norm(nodes[2] - nodes[3])
    dist_3_4 = np.linalg.norm(nodes[3] - nodes[4])
    dist_5_6 = np.linalg.norm(nodes[5] - nodes[6])
    dist_7_8 = np.linalg.norm(nodes[7] - nodes[8])
    dist_7_10 = np.linalg.norm(nodes[7] - nodes[10])
    dist_8_9 = np.linalg.norm(nodes[8] - nodes[9])

    dist_s0_4 = np.linalg.norm(vertices[0] - nodes[4])
    dist_s0_5 = np.linalg.norm(vertices[0] - nodes[5])
    dist_s0_9 = np.linalg.norm(vertices[0] - nodes[9])

    dist_s1_9 = np.linalg.norm(vertices[1] - nodes[9])
    dist_s1_1 = np.linalg.norm(vertices[1] - nodes[1])

    anchors_gt = np.array([[4, 3, 2, 1, 0, 5, 6, 9],
                           [9, 8, 7, 10, 1, 2, 3, 4]], dtype=np.int32)

    sp_dists_s0 = np.array([
        dist_s0_4,  # anchor node: 4
        dist_s0_4 + dist_3_4,  # anchor node: 3
        dist_s0_4 + dist_3_4 + dist_2_3,  # anchor node: 2
        dist_s0_4 + dist_3_4 + dist_2_3 + dist_1_2,  # anchor node: 1
        dist_s0_4 + dist_0_4,  # anchor node: 0
        dist_s0_5,  # anchor node: 5
        dist_s0_5 + dist_5_6,  # anchor node: 6
        dist_s0_9,  # anchor node: 9
    ])

    sp_dists_s1 = np.array([
        dist_s1_9,  # anchor node: 9
        dist_s1_9 + dist_8_9,  # anchor node: 8
        dist_s1_9 + dist_8_9 + dist_7_8,  # anchor node: 7
        dist_s1_9 + dist_8_9 + dist_7_8 + dist_7_10,  # anchor node: 10
        dist_s1_1,  # anchor node: 1
        dist_s1_1 + dist_1_2,  # anchor node: 2
        dist_s1_1 + dist_1_2 + dist_2_3,  # anchor node: 3
        dist_s1_1 + dist_1_2 + dist_3_4,  # anchor node: 4
    ])

    # @formatter:on
    def compute_anchor_weight(dist, _node_coverage):
        return math.exp(-(math.pow(dist, 2.0) / 2.0 * math.pow(_node_coverage, 2.0)))

    u_compute_anchor_weights = np.vectorize(compute_anchor_weight)
    weights_s0_gt = u_compute_anchor_weights(sp_dists_s0, node_coverage)
    weight_sum = weights_s0_gt.sum()
    weights_s0_gt /= weight_sum

    weights_s1_gt = u_compute_anchor_weights(sp_dists_s1, node_coverage)
    weight_sum = weights_s1_gt.sum()
    weights_s1_gt /= weight_sum
    return vertices, nodes, edges, anchors_gt, weights_s0_gt, weights_s1_gt


# see tests/test_data/AnchorComputationTest1.jpg for reference
def test_shortest_path_anchors_legacy():
    anchor_count = 8
    node_coverage = 1.0

    vertices, nodes, edges, anchors_gt, weights_s0_gt, weights_s1_gt = prepare_test_data_anchor_computation_manual(
        node_coverage)

    anchors, weights = nnrt.compute_vertex_anchors_shortest_path(vertices, nodes, edges, anchor_count, node_coverage)

    assert np.alltrue(anchors_gt == anchors)
    assert np.allclose(weights.sum(axis=1), [[1.0], [1.0]])
    assert np.allclose(weights[0], weights_s0_gt)
    assert np.allclose(weights[1], weights_s1_gt)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_shortest_path_anchors(device: o3d.core.Device):
    anchor_count = 8
    node_coverage = 1.0
    vertices, nodes, edges, anchors_gt, weights_s0_gt, weights_s1_gt = prepare_test_data_anchor_computation_manual(
        node_coverage)

    vertices_o3d = o3c.Tensor(vertices, device=device)

    nodes_o3d = o3c.Tensor(nodes, device=device)
    edges_o3d = o3c.Tensor(edges, device=device)

    anchors, weights = nnrt.geometry.compute_anchors_and_weights_shortest_path(
        vertices_o3d, nodes_o3d, edges_o3d, anchor_count, node_coverage)

    assert np.alltrue(anchors_gt == anchors.cpu().numpy())
    assert np.allclose(weights.cpu().numpy().sum(axis=1), [[1.0], [1.0]])
    assert np.allclose(weights.cpu().numpy()[0], weights_s0_gt)
    assert np.allclose(weights.cpu().numpy()[1], weights_s1_gt)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_compute_anchors_shortest_path_nnrtf_vs_legacy_nnrt_cpp(device):
    test_path = Path(__file__).parent.resolve()
    test_data_path = test_path / "test_data"
    anchor_test_data_dir = test_data_path / "anchor_assignment_and_weighing"
    depth_image_open3d_legacy = o3d.io.read_image(str(anchor_test_data_dir / "seq070_depth_000000.png"))
    depth_image_np = np.array(depth_image_open3d_legacy)
    mask_image_open3d_legacy = o3d.io.read_image(str(anchor_test_data_dir / "seq070_mask_000000_adult0.png"))
    mask_image_np = np.array(mask_image_open3d_legacy)
    depth_image_np[mask_image_np == 0] = 0
    fx, fy, cx, cy = camera.load_intrinsic_matrix_entries_from_text_4x4_matrix(
        str(anchor_test_data_dir / "seq070_intrinsics.txt"))

    point_image = image_processing.backproject_depth(depth_image_np, fx, fy, cx, cy, depth_scale=1000.0)  # (h, w, 3)
    node_coverage = 0.05
    anchor_count = 4
    vertices, vertex_pixels, faces = nnrt.compute_mesh_from_depth(point_image, 0.05)
    non_eroded_vertices = nnrt.get_vertex_erosion_mask(
        vertices, faces, 10, 4
    )

    node_coords, node_indices = nnrt.sample_nodes(
        vertices, non_eroded_vertices,
        node_coverage, True, False
    )

    graph_edges, graph_edge_weights, graph_edges_distances, node_to_vertex_distances = \
        nnrt.compute_edges_shortest_path(
            vertices, faces, node_indices,
            4, node_coverage, True
        )

    nodes = vertices[node_indices.flatten()]
    nodes_o3d = o3c.Tensor(nodes, device=device)
    vertices_o3d = o3c.Tensor(vertices, device=device)
    edges_o3d = o3c.Tensor(graph_edges, device=device)

    legacy_vertex_anchors, legacy_vertex_weights = \
        nnrt.compute_vertex_anchors_shortest_path(vertices, nodes, graph_edges, anchor_count, node_coverage)
    legacy_vertex_anchors_sorted = np.sort(legacy_vertex_anchors, axis=1)
    legacy_vertex_weights_sorted = np.sort(legacy_vertex_weights, axis=1)
    vertex_anchors, vertex_weights = \
        nnrt.geometry.compute_anchors_and_weights_shortest_path(vertices_o3d, nodes_o3d, edges_o3d,
                                                                anchor_count, node_coverage)
    vertex_anchors_sorted = np.sort(vertex_anchors.cpu().numpy(), axis=1)
    vertex_weights_sorted = np.sort(vertex_weights.cpu().numpy(), axis=1)

    assert np.allclose(vertex_anchors_sorted, legacy_vertex_anchors_sorted)
    assert np.allclose(vertex_weights_sorted, legacy_vertex_weights_sorted)


if __name__ == "__main__":
    test_compute_anchors_shortest_path_nnrtf_vs_legacy_nnrt_cpp(o3c.Device("CPU:0"))
