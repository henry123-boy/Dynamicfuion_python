import numpy as np


def compute_distance_matrix(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    Compute P x N distance matrix where P is the number of points and N is the number of nodes.
    :param points:
    :param nodes:
    :return:
    """
    nodes_tiled = np.tile(nodes, (len(points), 1, 1))
    points_tiled = np.tile(points, (1, len(nodes))).reshape(len(points), -1, 3)
    distance_matrix = np.linalg.norm(points_tiled - nodes_tiled, axis=2)
    return distance_matrix
