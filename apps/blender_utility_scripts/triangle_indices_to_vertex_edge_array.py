import numpy as np


def triangle_indices_to_vertex_edge_array(triangle_indices: np.ndarray, max_degree: int = 4) -> np.ndarray:
    vertex_indices = np.unique(triangle_indices)
    vertex_edge_array = -np.ones((vertex_indices.size, max_degree), dtype=np.int64)
    vertex_edge_sets = [set() for _ in vertex_indices]
    for triangle in triangle_indices:
        v0 = triangle[0]
        v1 = triangle[1]
        v2 = triangle[2]
        vertex_edge_sets[v0].add(v1)
        vertex_edge_sets[v0].add(v2)
        vertex_edge_sets[v1].add(v2)
    for source_vertex, edge_set in enumerate(vertex_edge_sets):
        sorted_edge_set = sorted(list(edge_set))
        for i_target_vertex, target_vertex in enumerate(sorted_edge_set):
            vertex_edge_array[source_vertex, i_target_vertex] = target_vertex
    return vertex_edge_array
