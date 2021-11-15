import numpy as np


def cpu_compute_normal(vertex_map):
    height, width = vertex_map.shape[:2]
    dv = ((vertex_map[2:, :] - vertex_map[:-2, :])[:, 1:-1])
    du = ((vertex_map[:, 2:] - vertex_map[:, :-2])[1:-1, :])

    dv = dv.reshape(-1, 3)
    du = du.reshape(-1, 3)
    normals = np.cross(du, dv)
    normals /= np.tile(np.linalg.norm(normals, axis=1).reshape(-1, 1), (1,3))

    normals[normals[:, 2] > 0] = -normals[normals[:, 2] > 0]
    normals = normals.reshape(height - 2, width - 2, -1)

    mask_y = np.logical_or(vertex_map[2:, :] == 0, vertex_map[:-2, :] == 0)[:, 1:-1]
    mask_x = np.logical_or(vertex_map[:, 2:] == 0, vertex_map[:, :-2] == 0)[1:-1, :]
    mask = np.logical_or(mask_x, mask_y)
    normals[mask] = 0

    normals_bordered = np.zeros_like(vertex_map)
    normals_bordered[1:height - 1, 1:width - 1] = normals

    return normals_bordered
