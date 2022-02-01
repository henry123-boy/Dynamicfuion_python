import typing

from deprecated import deprecated
import numpy as np
import open3d as o3d
import open3d.core as o3c
import nnrt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import cm
from telemetry.visualization.geometry.make_plane import make_z_aligned_image_plane

from nnrt.geometry import GraphWarpField as GraphWarpFieldOpen3DNative


def knn_edges_column_to_lines(node_edges: np.ndarray, neighbor_index) -> np.ndarray:
    lines = []
    for node_index in range(0, len(node_edges)):
        neighbor_node_index = node_edges[node_index, neighbor_index]
        if neighbor_node_index != -1:
            lines.append((node_index, neighbor_node_index))
    return np.array(lines)


def knn_graph_to_line_set(node_positions: np.ndarray, node_edges: np.ndarray,
                          clusters: np.ndarray = None) -> o3d.geometry.LineSet:
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
        plane_mesh = make_z_aligned_image_plane((extent_min[0], extent_min[1]), (extent_max[0], extent_max[1]), plane_z,
                                                background_image)
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


class GraphWarpFieldNumpy:
    def __init__(self, canonical_node_positions: np.ndarray, edges: np.ndarray, edge_weights: np.ndarray,
                 clusters: np.ndarray):
        self.nodes = canonical_node_positions
        self.edges = edges
        self.edge_weights = edge_weights
        self.clusters = clusters
        self.rotations = np.array([np.eye(3, dtype=np.float32)] * len(self.nodes))
        self.translations = np.zeros_like(self.nodes)

    def get_warped_nodes(self) -> np.ndarray:
        return self.nodes + self.translations

    def get_node_extent(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    @deprecated(version="1.0.0",
                reason="Will be phased out eventually during transition to native GraphWarpField class")
    def as_line_set_canonical(self) -> o3d.geometry.LineSet:
        return knn_graph_to_line_set(self.nodes, self.edges, self.clusters)

    def warp_mesh(self, mesh: o3d.geometry.TriangleMesh, node_coverage) -> o3d.geometry.TriangleMesh:
        vertices = np.array(mesh.vertices)
        vertex_anchors, vertex_weights = nnrt.compute_vertex_anchors_euclidean(self.nodes, vertices, node_coverage)
        i_vertex = 0
        deformed_vertices = np.zeros_like(vertices)
        for vertex in vertices:
            vertex_anchor_weights = vertex_weights[i_vertex]
            vertex_anchor_rotations = [self.rotations[anchor_node_index] for anchor_node_index in
                                       vertex_anchors[i_vertex]]
            vertex_anchor_translations = [self.translations[anchor_node_index] for anchor_node_index in
                                          vertex_anchors[i_vertex]]
            vertex_anchor_nodes = [self.nodes[anchor_node_index] for anchor_node_index in vertex_anchors[i_vertex]]
            deformed_vertex = np.zeros((3,), dtype=np.float32)
            for weight, rotation, translation, node in \
                    zip(vertex_anchor_weights, vertex_anchor_rotations, vertex_anchor_translations,
                        vertex_anchor_nodes):
                deformed_vertex += weight * (node + rotation.dot(vertex - node) + translation)
            deformed_vertices[i_vertex] = deformed_vertex
            i_vertex += 1

        mesh_warped = o3d.geometry.TriangleMesh(o3d.cuda.pybind.utility.Vector3dVector(deformed_vertices),
                                                mesh.triangles)
        mesh_warped.vertex_colors = mesh.vertex_colors
        mesh_warped.compute_vertex_normals()
        return mesh_warped

    def dump_to_disk(self, path: str) -> None:
        """
        Save the current graph to disk.
        :param path: where to save
        """
        np.savez(path, nodes=self.nodes, edges=self.edges, edge_weights=self.edge_weights, clusters=self.clusters,
                 rotations_mat=self.rotations, translations_vec=self.translations)


class GraphWarpFieldOpen3DPythonic:
    def __init__(self, nodes: o3c.Tensor, edges: o3c.Tensor, edge_weights: o3c.Tensor,
                 clusters: o3c.Tensor):
        self.nodes = nodes
        self.edges = edges
        self.edge_weights = edge_weights
        self.clusters = clusters
        self.rotations = o3c.Tensor(np.array([np.eye(3, dtype=np.float32)] * len(self.nodes)), device=nodes.device)
        self.translations = o3c.Tensor.zeros(self.nodes.shape, dtype=o3c.Dtype.Float32, device=nodes.device)

    def get_warped_nodes(self) -> o3c.Tensor:
        return self.nodes + self.translations

    def get_node_extent(self) -> typing.Tuple[o3c.Tensor, o3c.Tensor]:
        return self.nodes.min(dim=0), self.nodes.max(dim=0)

    def as_line_set_canonical(self) -> o3d.geometry.LineSet:
        return knn_graph_to_line_set(self.nodes.numpy(), self.edges.numpy(), self.clusters.numpy())

    def warp_mesh(self, mesh: o3d.t.geometry.TriangleMesh, node_coverage,
                  anchor_count=4) -> o3d.t.geometry.TriangleMesh:
        return nnrt.geometry.warp_triangle_mesh(mesh, self.nodes, self.rotations, self.translations, anchor_count,
                                                node_coverage)


def load_numpy_warp_field_from_disk(path: str) -> GraphWarpFieldNumpy:
    container = np.load(path)
    graph = GraphWarpFieldNumpy(container["nodes"], container["edges"], container["edge_weights"],
                                container["clusters"])
    graph.rotations = container["rotations"]
    graph.translations = container["translations"]
    return graph


def build_deformation_graph_from_mesh(mesh: o3d.t.geometry.TriangleMesh, node_coverage: float = 0.05,
                                      erosion_iteration_count: int = 10, erosion_min_neighbor_count: int = 4,
                                      neighbor_count: int = 8,
                                      minimum_valid_anchor_count: int = 3) -> GraphWarpFieldOpen3DNative:
    vertex_positions = np.array(mesh.vertices)
    triangle_vertex_indices = np.array(mesh.triangles)

    # === Build deformation graph ===

    erosion_mask = nnrt.get_vertex_erosion_mask(vertex_positions, triangle_vertex_indices, erosion_iteration_count,
                                                erosion_min_neighbor_count)
    nodes, node_vertex_indices = \
        nnrt.sample_nodes(vertex_positions, erosion_mask, node_coverage, use_only_non_eroded_indices=True,
                          random_shuffle=False)
    node_count = nodes.shape[0]

    edges, edge_weights, graph_edge_distances, node_to_vertex_distances = \
        nnrt.compute_edges_shortest_path(vertex_positions, triangle_vertex_indices, node_vertex_indices,
                                         neighbor_count, node_coverage, True)

    # ===== Remove nodes with not enough neighbors ===
    # TODO: break up the routines in create_graph_data.py and reuse them here & in the corresponding code below, then port to C++
    # valid_nodes_mask = np.ones((node_count, 1), dtype=bool)
    # # Mark nodes with not enough neighbors
    # nnrt.node_and_edge_clean_up(graph_edges, valid_nodes_mask)
    # # Get the list of invalid nodes
    # node_id_black_list = np.where(valid_nodes_mask is False)[0].tolist()
    #
    # # Get only valid nodes and their corresponding info
    # nodes = nodes[valid_nodes_mask.squeeze()]
    # node_vertex_indices = node_vertex_indices[valid_nodes_mask.squeeze()]
    # graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    # graph_edge_weights = graph_edge_weights[valid_nodes_mask.squeeze()]
    # graph_edge_distances = graph_edge_distances[valid_nodes_mask.squeeze()]
    #
    # #########################################################################
    # # Graph checks.
    # #########################################################################
    # num_nodes = nodes.shape[0]
    #
    # # Update node ids only if we actually removed nodes
    # if len(node_id_black_list) > 0:
    #     # 1. Mapping old indices to new indices
    #     count = 0
    #     node_id_mapping = {}
    #     for i, is_node_valid in enumerate(valid_nodes_mask):
    #         if not is_node_valid:
    #             node_id_mapping[i] = -1
    #         else:
    #             node_id_mapping[i] = count
    #             count += 1
    #
    #     # 2. Update graph_edges using the id mapping
    #     for node_id, graph_edge in enumerate(graph_edges):
    #         # compute mask of valid neighbors
    #         valid_neighboring_nodes = np.invert(
    #             np.isin(graph_edge, node_id_black_list))
    #
    #         # make a copy of the current neighbors' ids
    #         graph_edge_copy = np.copy(graph_edge)
    #         graph_edge_weights_copy = np.copy(graph_edge_weights[node_id])
    #         graph_edge_distances_copy = np.copy(graph_edge_distances[node_id])
    #
    #         # set the neighbors' ids to -1
    #         graph_edges[node_id] = -np.ones_like(graph_edge_copy)
    #         graph_edge_weights[node_id] = np.zeros_like(
    #             graph_edge_weights_copy)
    #         graph_edge_distances[node_id] = np.zeros_like(
    #             graph_edge_distances_copy)
    #
    #         count_valid_neighbors = 0
    #         for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
    #             if is_valid_neighbor:
    #                 # current neighbor id
    #                 current_neighbor_id = graph_edge_copy[neighbor_idx]
    #
    #                 # get mapped neighbor id
    #                 if current_neighbor_id == -1:
    #                     mapped_neighbor_id = -1
    #                 else:
    #                     mapped_neighbor_id = node_id_mapping[current_neighbor_id]
    #
    #                 graph_edges[node_id,
    #                             count_valid_neighbors] = mapped_neighbor_id
    #                 graph_edge_weights[node_id,
    #                                     count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
    #                 graph_edge_distances[node_id,
    #                                       count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]
    #
    #                 count_valid_neighbors += 1
    #
    #         # normalize edges' weights
    #         sum_weights = np.sum(graph_edge_weights[node_id])
    #         if sum_weights > 0:
    #             graph_edge_weights[node_id] /= sum_weights
    #         else:
    #             print("Hmmmmm", graph_edge_weights[node_id])
    #             raise Exception("Not good")

    #########################################################################
    # Compute clusters.
    #########################################################################
    clusters_size_list, clusters = nnrt.compute_clusters(edges)
    for i, cluster_size in enumerate(clusters_size_list):
        if cluster_size <= 2:
            print("Cluster is too small {}".format(clusters_size_list))
            print("It only has nodes:", np.where(clusters == i)[0])

    nodes_o3d = o3c.Tensor(nodes, device=mesh.device)
    edges_o3d = o3c.Tensor(edges, device=mesh.device)
    edge_weights_o3d = o3c.Tensor(edge_weights, device=mesh.device)
    clusters_o3d = o3c.Tensor(clusters, device=mesh.device)

    return GraphWarpFieldOpen3DNative(nodes_o3d, edges_o3d, edge_weights_o3d, clusters_o3d, node_coverage=node_coverage,
                                      threshold_nodes_by_distance=minimum_valid_anchor_count > 0,
                                      minimum_valid_anchor_count=minimum_valid_anchor_count)


def draw_deformation_graph(deformation_graph: GraphWarpFieldNumpy,
                           background_image: o3d.geometry.Image = None) -> None:
    line_set = deformation_graph.as_line_set_canonical()

    extent_min, extent_max = deformation_graph.get_node_extent()
    plane_z = extent_max[2]

    if background_image is not None:
        plane_mesh = make_z_aligned_image_plane((extent_min[0], extent_min[1]), (extent_max[0], extent_max[1]), plane_z,
                                                background_image)
        geometries = [plane_mesh, line_set]
    else:
        geometries = [line_set]

    o3d.visualization.draw_geometries(geometries,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)
