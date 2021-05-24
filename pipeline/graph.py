import dq3d
import numpy as np
import open3d as o3d
import nnrt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import cm
from utils.viz.primitive import make_z_aligned_image_plane
from data import camera
from dq3d import dualquat, quat, op


def knn_edges_column_to_lines(node_edges: np.ndarray, neighbor_index) -> np.ndarray:
    lines = []
    for node_index in range(0, len(node_edges)):
        neighbor_node_index = node_edges[node_index, neighbor_index]
        if neighbor_node_index != -1:
            lines.append((node_index, neighbor_node_index))
    return np.array(lines)


def knn_graph_to_line_set(node_positions: np.ndarray, node_edges: np.ndarray, clusters: np.ndarray = None) -> o3d.geometry.LineSet:
    first_connections = node_edges[:, :1].copy()
    node_indices = np.arange(0, node_positions.shape[0]).reshape(-1, 1)
    lines_0 = np.concatenate((node_indices.copy(), first_connections), axis=1)
    neighbor_line_source_sets = [lines_0]
    for neighbor_index in range(1, node_edges.shape[1]):
        neighbor_line_set = knn_edges_column_to_lines(node_edges, neighbor_index)
        if len(neighbor_line_set.shape) == 2:
            neighbor_line_source_sets.append(neighbor_line_set)

    lines = np.concatenate(neighbor_line_source_sets, axis=0)

    if clusters is None:
        colors = [[1, 0, 0] for _ in range(len(lines))]
    else:
        colors = []
        for line in lines:
            color = cm.tab10(clusters[line[0]])
            colors.append([color[0], color[1], color[2]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(node_positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_knn_graph(node_positions: np.ndarray, node_edges: np.ndarray, clusters: np.ndarray = None,
                   background_image: o3d.geometry.Image = None) -> None:
    line_set = knn_graph_to_line_set(node_positions, node_edges, clusters)

    extent_max = node_positions.max(axis=0)
    extent_min = node_positions.min(axis=0)
    plane_z = extent_max[2]

    if background_image is not None:
        plane_mesh = make_z_aligned_image_plane((extent_min[0], extent_min[1]), (extent_max[0], extent_max[1]), plane_z, background_image)
        geometries = [plane_mesh, line_set]
    else:
        geometries = [line_set]

    o3d.visualization.draw_geometries(geometries,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)


def find_knn_graph_connected_components(knn_edges: np.ndarray) -> np.ndarray:
    source_node_index = 0
    edge_matrix = lil_matrix((knn_edges.shape[0], knn_edges.shape[0]), dtype=int)
    for destination_nodes in knn_edges:
        for destination_node_index in destination_nodes:
            if destination_node_index != -1:
                edge_matrix[source_node_index, destination_node_index] = 1
        source_node_index += 1

    edge_matrix_compressed = csr_matrix(edge_matrix)
    _, labels = connected_components(csgraph=edge_matrix_compressed, directed=False, return_labels=True)
    return labels


class DeformationGraph:
    def __init__(self, canonical_node_positions: np.ndarray, edges: np.ndarray, edge_weights: np.ndarray,
                 clusters: np.ndarray):
        self.nodes = canonical_node_positions
        self.edges = edges
        self.edge_weights = edge_weights
        self.clusters = clusters
        self.transformations = [dualquat(quat.identity())] * len(self.nodes)

    def get_extent_canonical(self):
        return self.nodes.max(axis=0), self.nodes.min(axis=0)

    def as_line_set_canonical(self):
        return knn_graph_to_line_set(self.nodes, self.edges, self.clusters)

    def warp_mesh(self, mesh: o3d.geometry.TriangleMesh, node_coverage) -> o3d.geometry.TriangleMesh:
        # TODO: provide an equivalent routine for o3d.t.geometry.TriangleMesh on CUDA, so that we don't have to convert
        #  to legacy mesh at all
        vertices = np.array(mesh.vertices)
        vertex_anchors, vertex_weights = nnrt.compute_vertex_anchors_euclidean(self.nodes, vertices, node_coverage)
        i_vertex = 0
        deformed_vertices = np.zeros_like(vertices)
        for vertex in vertices:
            vertex_anchor_quaternions = [self.transformations[anchor_node_index] for anchor_node_index in vertex_anchors[i_vertex]]
            vertex_anchor_weights = vertex_weights[i_vertex]
            deformed_vertices[i_vertex] = op.dlb(vertex_anchor_weights, vertex_anchor_quaternions).transform_point(vertex)
            i_vertex += 1

        mesh_warped = o3d.geometry.TriangleMesh(o3d.cuda.pybind.utility.Vector3dVector(deformed_vertices), mesh.triangles)
        mesh_warped.vertex_colors = mesh.vertex_colors
        return mesh_warped


def build_deformation_graph_from_mesh(mesh: o3d.geometry.TriangleMesh, node_coverage: float = 0.05,
                                      erosion_iteration_count: int = 10, erosion_min_neighbor_count: int = 4,
                                      neighbor_count: int = 8) -> DeformationGraph:
    vertex_positions = np.array(mesh.vertices)
    triangle_vertex_indices = np.array(mesh.triangles)

    # === Build deformation graph ===

    erosion_mask = nnrt.get_vertex_erosion_mask(vertex_positions, triangle_vertex_indices, erosion_iteration_count, erosion_min_neighbor_count)
    nodes, node_vertex_indices = \
        nnrt.sample_nodes(vertex_positions, erosion_mask, node_coverage, use_only_non_eroded_indices=True, random_shuffle=False)

    graph_edges, graph_edge_weights, graph_edge_distances, node_to_vertex_distances = \
        nnrt.compute_edges_geodesic(vertex_positions, triangle_vertex_indices, node_vertex_indices,
                                    neighbor_count, node_coverage, True)

    # TODO: do the cleanup properly in a separate routine, follow logic from create_graph_data.py
    # node_count = node_positions.shape[0]
    # valid_nodes_mask = np.ones((node_count, 1), dtype=bool)
    # # Mark nodes with not enough neighbors
    # nnrt.node_and_edge_clean_up(graph_edges, valid_nodes_mask)
    # # Get the list of invalid nodes
    # node_id_black_list = np.where(valid_nodes_mask is False)[0].tolist()

    cluster_sizes, graph_clusters = nnrt.compute_clusters(graph_edges)

    return DeformationGraph(nodes, graph_edges, graph_edge_weights, graph_clusters)


def build_deformation_graph_from_depth_image(depth_image: np.ndarray,
                                             intrinsics: o3d.camera.PinholeCameraIntrinsic,
                                             downsampling_factor: int) -> DeformationGraph:
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics)

    point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)

    node_positions, edges, pixel_anchors, pixel_weights = \
        nnrt.construct_regular_graph(point_image,
                                     depth_image.shape[1] // downsampling_factor,
                                     depth_image.shape[0] // downsampling_factor,
                                     2.0, 0.05, 3.0)
    clusters = find_knn_graph_connected_components(knn_edges=edges)
    return DeformationGraph(node_positions, edges, np.array([1] * len(edges)), clusters)


def draw_deformation_graph(deformation_graph: DeformationGraph,
                           background_image: o3d.geometry.Image = None) -> None:
    line_set = deformation_graph.as_line_set_canonical()

    extent_max, extent_min = deformation_graph.get_extent_canonical()
    plane_z = extent_max[2]

    if background_image is not None:
        plane_mesh = make_z_aligned_image_plane((extent_min[0], extent_min[1]), (extent_max[0], extent_max[1]), plane_z, background_image)
        geometries = [plane_mesh, line_set]
    else:
        geometries = [line_set]

    o3d.visualization.draw_geometries(geometries,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)
