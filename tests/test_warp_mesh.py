import math
import open3d as o3d
import open3d.core as o3c
import pytest
import numpy as np
from warp_field.graph import DeformationGraphNumpy, DeformationGraphOpen3D


@pytest.fixture
def ground_truth_vertices() -> np.ndarray:

    vertices = np.array([[-2.29245767e-01, -1.41411602e-08, 2.67084956e-01],
                                      [6.91970110e-01, -1.42059857e-08, -7.78945014e-02],
                                      [1.16394386e-01, -1.43351269e-08, 1.19694209e+00],
                                      [1.04608846e+00, -1.49011612e-08, 8.31702471e-01],
                                      [2.23095283e-01, 1.00000012e+00, -7.21708238e-02],
                                      [1.13396776e+00, 9.99999940e-01, 1.43104389e-01],
                                      [4.56829136e-03, 9.99999940e-01, 8.61463904e-01],
                                      [9.16629255e-01, 1.00000000e+00, 1.09062088e+00],
                                      [5.91987669e-01, 1.00000012e+00, 4.94710982e-01],
                                      [1.04602730e+00, 1.00000000e+00, 6.24803603e-01],
                                      [6.72028065e-01, 9.99999940e-01, 4.03444134e-02],
                                      [1.07328296e-01, 1.00000000e+00, 3.99524152e-01],
                                      [4.31075335e-01, 1.00000000e+00, 9.46507633e-01],
                                      [-5.95213622e-02, -1.42774095e-08, 7.33223557e-01],
                                      [2.05226243e-03, 4.99999970e-01, 6.10076189e-01],
                                      [-8.64362046e-02, 5.00000000e-01, 1.59977779e-01],
                                      [7.47523531e-02, 5.00000000e-01, 1.07201600e+00],
                                      [2.22478822e-01, -1.43897658e-08, 9.65380520e-02],
                                      [3.92807037e-01, -1.49011612e-08, 5.61104476e-01],
                                      [8.54746759e-01, -1.49011612e-08, 3.69762778e-01],
                                      [5.84148765e-01, -1.49011612e-08, 1.02304411e+00],
                                      [8.53231609e-01, 4.99999970e-01, 2.73631583e-03],
                                      [9.07589078e-01, 4.99999940e-01, 4.40219164e-01],
                                      [1.01334465e+00, 5.00000000e-01, 8.97190273e-01],
                                      [5.57375371e-01, 5.00000000e-01, 1.00965750e+00],
                                      [3.91291916e-01, 5.00000000e-01, 7.54364058e-02]], dtype=np.float32)
    return vertices


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_warp_mesh_numpy_mat(device, ground_truth_vertices):
    mesh = o3d.geometry.TriangleMesh.create_box()
    # increase number of iterations for a "pretty" twisted box mesh (test will fail, of course, due to wrong GT data)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)

    mesh.compute_vertex_normals()
    nodes = np.array([
        [0., 0., 0.],  # bottom
        [1., 0., 0.],  # bottom
        [0., 0., 1.],  # bottom
        [1., 0., 1.],  # bottom
        [0., 1., 0.],  # top
        [1., 1., 0.],  # top
        [0., 1., 1.],  # top
        [1., 1., 1.]  # top
    ], dtype=np.float32)

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
    global_rotation_matrix_top = np.array([[math.cos(mesh_rotation_angle), 0.0, math.sin(mesh_rotation_angle)],
                                           [0., 1., 0.],
                                           [-math.sin(mesh_rotation_angle), 0.0, math.cos(mesh_rotation_angle)]],
                                          dtype=np.float32)
    global_rotation_matrix_bottom = np.array([[math.cos(-mesh_rotation_angle), 0.0, math.sin(-mesh_rotation_angle)],
                                              [0., 1., 0.],
                                              [-math.sin(-mesh_rotation_angle), 0.0, math.cos(-mesh_rotation_angle)]],
                                             dtype=np.float32)

    nodes_center = nodes.mean(axis=0)
    nodes_rotated = nodes_center + np.concatenate(((nodes[:4] - nodes_center).dot(global_rotation_matrix_bottom),
                                                   (nodes[4:] - nodes_center).dot(global_rotation_matrix_top)), axis=0)

    # graph_numpy.rotations_mat = np.stack([global_rotation_matrix_bottom] * 4 + [global_rotation_matrix_top] * 4)
    # TODO why should rotations be flipped??? Should they be inverted during warping?
    graph_numpy.rotations_mat = np.stack([global_rotation_matrix_top] * 4 + [global_rotation_matrix_bottom] * 4)
    for i_node, (node, node_rotated) in enumerate(zip(nodes, nodes_rotated)):
        graph_numpy.translations_vec[i_node] = node_rotated - node

    warped_mesh = graph_numpy.warp_mesh_mat(mesh, 0.5)

    visualize_results = False
    if visualize_results:
        o3d.visualization.draw_geometries([warped_mesh],
                                          zoom=0.8,
                                          front=[0.0, 0., 2.],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 1, 0])
    assert np.allclose(np.array(warped_mesh.vertices), ground_truth_vertices, atol=1e-6)


@pytest.mark.parametrize("device", [o3d.core.Device('cuda:0'), o3d.core.Device('cpu:0')])
def test_warp_mesh_open3d_mat(device, ground_truth_vertices):
    mesh_legacy = o3d.geometry.TriangleMesh.create_box()
    mesh_legacy = mesh_legacy.subdivide_midpoint(number_of_iterations=1)
    mesh_legacy.compute_vertex_normals()
    nodes = np.array([
        [0., 0., 0.],  # bottom
        [1., 0., 0.],  # bottom
        [0., 0., 1.],  # bottom
        [1., 0., 1.],  # bottom
        [0., 1., 0.],  # top
        [1., 1., 0.],  # top
        [0., 1., 1.],  # top
        [1., 1., 1.]  # top
    ], dtype=np.float32)
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
    global_rotation_matrix_top = np.array([[math.cos(mesh_rotation_angle), 0.0, math.sin(mesh_rotation_angle)],
                                           [0., 1., 0.],
                                           [-math.sin(mesh_rotation_angle), 0.0, math.cos(mesh_rotation_angle)]],
                                          dtype=np.float32)
    global_rotation_matrix_bottom = np.array([[math.cos(-mesh_rotation_angle), 0.0, math.sin(-mesh_rotation_angle)],
                                              [0., 1., 0.],
                                              [-math.sin(-mesh_rotation_angle), 0.0, math.cos(-mesh_rotation_angle)]],
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

    visualize_results = False
    if visualize_results:
        o3d.visualization.draw_geometries([warped_mesh_legacy],
                                          zoom=0.8,
                                          front=[0.0, 0., 2.],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 1, 0])
    new_vertices = np.array(warped_mesh_legacy.vertices)

    assert np.allclose(new_vertices, ground_truth_vertices, atol=1e-6)
