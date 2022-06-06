import numpy as np
import open3d as o3d
import open3d.core as o3c
import nnrt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from nnrt.geometry import GraphWarpField


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


def build_deformation_graph_from_mesh(mesh: o3d.t.geometry.TriangleMesh, node_coverage: float = 0.05,
                                      erosion_iteration_count: int = 10, erosion_min_neighbor_count: int = 4,
                                      neighbor_count: int = 8,
                                      minimum_valid_anchor_count: int = 3) -> GraphWarpField:
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

    return GraphWarpField(nodes_o3d, edges_o3d, edge_weights_o3d, clusters_o3d, node_coverage=node_coverage,
                          threshold_nodes_by_distance=minimum_valid_anchor_count > 0,
                          minimum_valid_anchor_count=minimum_valid_anchor_count)
