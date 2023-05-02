import numpy as np


def triangle_indices_to_vertex_edge_array(triangle_indices: np.ndarray, max_degree: int = 4) -> np.ndarray:
    vertex_indices = np.unique(triangle_indices)
    vertex_edge_array = -np.ones((vertex_indices.size, max_degree), dtype=np.int64)
    vertex_edge_sets = [set() for _ in vertex_indices]

    def add_edge(vA, vB):
        if vA < vB:
            vertex_edge_sets[vA].add(vB)
        else:
            vertex_edge_sets[vB].add(vA)

    for triangle in triangle_indices:
        v0 = triangle[0]
        v1 = triangle[1]
        v2 = triangle[2]
        add_edge(v0, v1)
        add_edge(v1, v2)
        add_edge(v2, v0)

    for source_vertex, edge_set in enumerate(vertex_edge_sets):
        sorted_edge_set = sorted(list(edge_set))
        for i_target_vertex, target_vertex in enumerate(sorted_edge_set):
            vertex_edge_array[source_vertex, i_target_vertex] = target_vertex
    return vertex_edge_array


def radius_median_subsample_3d_points(source_points: np.array):
    pass
