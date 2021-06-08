import numpy as np
from sklearn.preprocessing import normalize


def cpu_compute_normal(vertex_map):
    height, width = vertex_map.shape[:2]
    dv = ((vertex_map[2:, :] - vertex_map[:-2, :])[:, 1:-1]).reshape(-1, 3)
    du = ((vertex_map[:, 2:] - vertex_map[:, :-2])[1:-1, :]).reshape(-1, 3)
    normals = normalize(np.cross(du, dv), axis=1)
    normals[normals[:, 2] > 0] = -normals[normals[:, 2] > 0]
    normals = normals.reshape(height-2, width-2, -1)
    normals_bordered = np.zeros_like(vertex_map)
    normals_bordered[1:height-1, 1:width-1] = normals
    return normals_bordered

