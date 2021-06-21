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
    nodes += np.array([0.1, 0.2, 0.3])
    nodes_o3d = o3c.Tensor(nodes, device=device)

    points = np.array(mesh.vertices)
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

    # @formatter:off
    ground_truth_vertices = np.array([[ 0.19256668,  0.        , -0.1164462 ],
                                      [ 1.11644602,  0.        ,  0.19256668],
                                      [-0.1164462 ,  0.        ,  0.80743313],
                                      [ 0.80743313,  0.        ,  1.11644602],
                                      [-0.1164462 ,  0.99999988,  0.19256668],
                                      [ 0.80743313,  0.99999988, -0.11644619],
                                      [ 0.19256668,  0.99999988,  1.11644614],
                                      [ 1.11644614,  0.99999988,  0.80743313],
                                      [ 0.5       ,  1.        ,  0.5       ],
                                      [ 0.96193969,  1.        ,  0.30865833],
                                      [ 0.30865833,  1.        ,  0.03806025],
                                      [ 0.03806025,  1.        ,  0.69134176],
                                      [ 0.69134176,  1.        ,  0.96193975],
                                      [ 0.03806025,  0.        ,  0.35427532],
                                      [ 0.03806025,  0.5       ,  0.5       ],
                                      [ 0.01525176,  0.5       ,  0.06086874],
                                      [ 0.06086874,  0.5       ,  0.98474824],
                                      [ 0.64572477,  0.        ,  0.03806025],
                                      [ 0.5       ,  0.        ,  0.5       ],
                                      [ 0.96193975,  0.        ,  0.64572477],
                                      [ 0.35427532,  0.        ,  0.96193969],
                                      [ 0.93913126,  0.5       ,  0.01525176],
                                      [ 0.96193975,  0.5       ,  0.5       ],
                                      [ 0.98474824,  0.5       ,  0.93913126],
                                      [ 0.5       ,  0.5       ,  0.96193975],
                                      [ 0.5       ,  0.5       ,  0.03806025]], dtype=np.float32)
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

    # # @formatter:off
    ground_truth_vertices = np.array([[ 0.19256668,  0.        , -0.1164462 ],
                                      [ 1.11644602,  0.        ,  0.19256668],
                                      [-0.1164462 ,  0.        ,  0.80743313],
                                      [ 0.80743313,  0.        ,  1.11644602],
                                      [-0.1164462 ,  0.99999988,  0.19256668],
                                      [ 0.80743313,  0.99999988, -0.11644619],
                                      [ 0.19256668,  0.99999988,  1.11644614],
                                      [ 1.11644614,  0.99999988,  0.80743313],
                                      [ 0.5       ,  1.        ,  0.5       ],
                                      [ 0.96193969,  1.        ,  0.30865833],
                                      [ 0.30865833,  1.        ,  0.03806025],
                                      [ 0.03806025,  1.        ,  0.69134176],
                                      [ 0.69134176,  1.        ,  0.96193975],
                                      [ 0.03806025,  0.        ,  0.35427532],
                                      [ 0.03806025,  0.5       ,  0.5       ],
                                      [ 0.01525176,  0.5       ,  0.06086874],
                                      [ 0.06086874,  0.5       ,  0.98474824],
                                      [ 0.64572477,  0.        ,  0.03806025],
                                      [ 0.5       ,  0.        ,  0.5       ],
                                      [ 0.96193975,  0.        ,  0.64572477],
                                      [ 0.35427532,  0.        ,  0.96193969],
                                      [ 0.93913126,  0.5       ,  0.01525176],
                                      [ 0.96193975,  0.5       ,  0.5       ],
                                      [ 0.98474824,  0.5       ,  0.93913126],
                                      [ 0.5       ,  0.5       ,  0.96193975],
                                      [ 0.5       ,  0.5       ,  0.03806025]], dtype=np.float32)
    # # @formatter:on
    visualize_results = False
    if visualize_results:
        o3d.visualization.draw_geometries([warped_mesh_legacy],
                                          zoom=0.8,
                                          front=[0.0, 0., 2.],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 1, 0])
    new_vertices = np.array(warped_mesh_legacy.vertices)
    mismatches = np.where(np.bitwise_not(np.isclose(new_vertices, ground_truth_vertices, atol=1e-4)))
    print(np.stack((new_vertices[mismatches], ground_truth_vertices[mismatches])).T)
    print(mismatches)
    print(new_vertices)

    assert np.allclose(new_vertices, ground_truth_vertices, atol=1e-6)
