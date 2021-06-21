import math
import open3d as o3d
import open3d.core as o3c
import nnrt
import pytest
import numpy as np
from pipeline.graph import DeformationGraphNumpy, DeformationGraphOpen3D


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_compute_anchors_euclidean(device):
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
    vertex_anchors, vertex_weights = nnrt.geometry.compute_anchors_and_weights_euclidean(points_o3d, nodes_o3d, anchor_count, 0, 0.5)
    vertex_anchors_sorted = np.sort(vertex_anchors.cpu().numpy(), axis=1)
    vertex_weights_sorted = np.sort(vertex_weights.cpu().numpy(), axis=1)

    assert np.allclose(vertex_anchors_gt, old_vertex_anchors_sorted)
    assert np.allclose(vertex_anchors_sorted, old_vertex_anchors_sorted)
    assert np.allclose(vertex_weights_sorted, old_vertex_weights_sorted)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_warp_mesh_numpy_mat(device):
    mesh = o3d.geometry.TriangleMesh.create_box()
    # increase number of iterations for a "pretty" twisted box mesh (test will fail, of course, due to wrong GT data)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)

    mesh.compute_vertex_normals()
    nodes = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.],
                      [1., 0., 1.],
                      [0., 1., 0.],
                      [1., 1., 0.],
                      [0., 1., 1.],
                      [1., 1., 1.]], dtype=np.float32)

    nodes += np.array([0.1, 0.2, 0.3])

    edges = np.array([[1, 2, 4],
                      [0, 5, 3],
                      [0, 3, 6],
                      [1, 2, 7],
                      [0, 5, 6],
                      [1, 4, 7],
                      [2, 4, 7],
                      [3, 5, 6]], dtype=np.int32)

    graph_numpy = DeformationGraphNumpy(nodes, edges, np.array([1] * len(edges), dtype=np.float32),
                                        np.array([0] * len(nodes), dtype=np.int32))

    mesh_rotation_angle = math.radians(22.5)
    global_rotation_matrix_top = np.array([[math.cos(mesh_rotation_angle), 0.0, -math.sin(mesh_rotation_angle)],
                                           [0., 1., 0.],
                                           [math.sin(mesh_rotation_angle), 0.0, math.cos(mesh_rotation_angle)]],
                                          dtype=np.float32)
    global_rotation_matrix_bottom = np.array([[math.cos(-mesh_rotation_angle), 0.0, -math.sin(-mesh_rotation_angle)],
                                              [0., 1., 0.],
                                              [math.sin(-mesh_rotation_angle), 0.0, math.cos(-mesh_rotation_angle)]],
                                             dtype=np.float32)

    nodes_center = nodes.mean(axis=0)
    nodes_rotated = nodes_center + np.concatenate(((nodes[:4] - nodes_center).dot(global_rotation_matrix_bottom),
                                                   (nodes[4:] - nodes_center).dot(global_rotation_matrix_top)), axis=0)

    graph_numpy.rotations_mat = np.stack([global_rotation_matrix_top] * 4 + [global_rotation_matrix_bottom] * 4)
    for i_node, (node, node_rotated) in enumerate(zip(nodes, nodes_rotated)):
        graph_numpy.translations_vec[i_node] = node_rotated - node

    warped_mesh = graph_numpy.warp_mesh_mat(mesh, 0.5)

    ground_truth_vertices = np.array([[3.20590377e-01, -1.41411602e-08, -1.45292148e-01],
                                      [1.24713349e+00, -1.42059857e-08, 1.99687317e-01],
                                      [-2.50497740e-02, -1.43351269e-08, 7.72609770e-01],
                                      [8.93015027e-01, -1.49011612e-08, 1.13784933e+00],
                                      [-1.31750658e-01, 1.00000012e+00, 1.93963617e-01],
                                      [8.05135846e-01, 9.99999940e-01, -2.13116035e-02],
                                      [8.67763162e-02, 9.99999940e-01, 1.10808790e+00],
                                      [1.02247429e+00, 1.00000000e+00, 8.78930807e-01],
                                      [4.23236459e-01, 1.00000012e+00, 5.50961316e-01],
                                      [8.93076181e-01, 1.00000000e+00, 4.20868665e-01],
                                      [3.43196094e-01, 9.99999940e-01, 8.14483836e-02],
                                      [-1.59836709e-02, 1.00000000e+00, 6.46148205e-01],
                                      [5.84148765e-01, 1.00000000e+00, 1.02304423e+00],
                                      [1.50865987e-01, -1.42774095e-08, 3.12448800e-01],
                                      [8.92923549e-02, 4.99999970e-01, 4.35596019e-01],
                                      [1.77780822e-01, 5.00000000e-01, -3.81849892e-02],
                                      [1.65922549e-02, 5.00000000e-01, 8.97535801e-01],
                                      [7.92745352e-01, -1.43897658e-08, 2.52547618e-02],
                                      [6.22417033e-01, -1.49011612e-08, 4.84567821e-01],
                                      [1.08435690e+00, -1.49011612e-08, 6.75909519e-01],
                                      [4.31075335e-01, -1.49011612e-08, 9.46507573e-01],
                                      [1.08587193e+00, 4.99999970e-01, 1.19056471e-01],
                                      [1.03151441e+00, 4.99999940e-01, 6.05453014e-01],
                                      [9.25759017e-01, 5.00000000e-01, 1.07236147e+00],
                                      [4.57848758e-01, 5.00000000e-01, 9.59894180e-01],
                                      [6.23932183e-01, 5.00000000e-01, 4.63563874e-02]], dtype=np.float32)
    visualize_results = False
    if visualize_results:
        o3d.visualization.draw_geometries([warped_mesh],
                                          zoom=0.8,
                                          front=[0.0, 0., 2.],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 1, 0])

    assert np.allclose(np.array(warped_mesh.vertices), ground_truth_vertices, atol=1e-6)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_warp_mesh_open3d_mat(device):
    mesh_legacy = o3d.geometry.TriangleMesh.create_box()
    mesh_legacy = mesh_legacy.subdivide_midpoint(number_of_iterations=1)
    mesh_legacy.compute_vertex_normals()
    nodes = np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.],
                      [1., 0., 1.],
                      [0., 1., 0.],
                      [1., 1., 0.],
                      [0., 1., 1.],
                      [1., 1., 1.]], dtype=np.float32)
    nodes += np.array([0.1, 0.2, 0.3])  # to make distances unique

    nodes_o3d = o3c.Tensor(nodes, device=device)

    edges = np.array([[1, 2, 4],
                      [0, 5, 3],
                      [0, 3, 6],
                      [1, 2, 7],
                      [0, 5, 6],
                      [1, 4, 7],
                      [2, 4, 7],
                      [3, 5, 6]], dtype=np.int32)
    edges_o3d = o3c.Tensor(edges, device=device)
    edge_weights_o3d = o3c.Tensor(np.array([1] * len(edges)), device=device)
    clusters_o3d = o3c.Tensor(np.array([0] * len(nodes)), device=device)

    graph_open3d = DeformationGraphOpen3D(nodes_o3d, edges_o3d, edge_weights_o3d, clusters_o3d)

    mesh_rotation_angle = math.radians(22.5)
    global_rotation_matrix_top = np.array([[math.cos(mesh_rotation_angle), 0.0, -math.sin(mesh_rotation_angle)],
                                           [0., 1., 0.],
                                           [math.sin(mesh_rotation_angle), 0.0, math.cos(mesh_rotation_angle)]],
                                          dtype=np.float32)
    global_rotation_matrix_bottom = np.array([[math.cos(-mesh_rotation_angle), 0.0, -math.sin(-mesh_rotation_angle)],
                                              [0., 1., 0.],
                                              [math.sin(-mesh_rotation_angle), 0.0, math.cos(-mesh_rotation_angle)]],
                                             dtype=np.float32)

    nodes_center = nodes.mean(axis=0)
    nodes_rotated = nodes_center + np.concatenate(((nodes[:4] - nodes_center).dot(global_rotation_matrix_bottom),
                                                   (nodes[4:] - nodes_center).dot(global_rotation_matrix_top)), axis=0)

    graph_open3d.rotations_mat = o3c.Tensor(np.stack([global_rotation_matrix_top] * 4 + [global_rotation_matrix_bottom] * 4),
                                            device=device)
    translations = nodes_rotated - nodes
    graph_open3d.translations_vec = o3c.Tensor(translations,
                                               device=device)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(mesh_legacy, device=device)

    warped_mesh = graph_open3d.warp_mesh_mat(mesh, 0.5)
    warped_mesh_legacy: o3d.geometry.TriangleMesh = warped_mesh.to_legacy_triangle_mesh()
    warped_mesh_legacy.compute_vertex_normals()

    ground_truth_vertices = np.array([[3.20590377e-01, -1.41411602e-08, -1.45292148e-01],
                                      [1.24713349e+00, -1.42059857e-08, 1.99687317e-01],
                                      [-2.50497740e-02, -1.43351269e-08, 7.72609770e-01],
                                      [8.93015027e-01, -1.49011612e-08, 1.13784933e+00],
                                      [-1.31750658e-01, 1.00000012e+00, 1.93963617e-01],
                                      [8.05135846e-01, 9.99999940e-01, -2.13116035e-02],
                                      [8.67763162e-02, 9.99999940e-01, 1.10808790e+00],
                                      [1.02247429e+00, 1.00000000e+00, 8.78930807e-01],
                                      [4.23236459e-01, 1.00000012e+00, 5.50961316e-01],
                                      [8.93076181e-01, 1.00000000e+00, 4.20868665e-01],
                                      [3.43196094e-01, 9.99999940e-01, 8.14483836e-02],
                                      [-1.59836709e-02, 1.00000000e+00, 6.46148205e-01],
                                      [5.84148765e-01, 1.00000000e+00, 1.02304423e+00],
                                      [1.50865987e-01, -1.42774095e-08, 3.12448800e-01],
                                      [8.92923549e-02, 4.99999970e-01, 4.35596019e-01],
                                      [1.77780822e-01, 5.00000000e-01, -3.81849892e-02],
                                      [1.65922549e-02, 5.00000000e-01, 8.97535801e-01],
                                      [7.92745352e-01, -1.43897658e-08, 2.52547618e-02],
                                      [6.22417033e-01, -1.49011612e-08, 4.84567821e-01],
                                      [1.08435690e+00, -1.49011612e-08, 6.75909519e-01],
                                      [4.31075335e-01, -1.49011612e-08, 9.46507573e-01],
                                      [1.08587193e+00, 4.99999970e-01, 1.19056471e-01],
                                      [1.03151441e+00, 4.99999940e-01, 6.05453014e-01],
                                      [9.25759017e-01, 5.00000000e-01, 1.07236147e+00],
                                      [4.57848758e-01, 5.00000000e-01, 9.59894180e-01],
                                      [6.23932183e-01, 5.00000000e-01, 4.63563874e-02]], dtype=np.float32)
    visualize_results = False
    if visualize_results:
        o3d.visualization.draw_geometries([warped_mesh_legacy],
                                          zoom=0.8,
                                          front=[0.0, 0., 2.],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 1, 0])
    new_vertices = np.array(warped_mesh_legacy.vertices)

    assert np.allclose(new_vertices, ground_truth_vertices, atol=1e-6)
