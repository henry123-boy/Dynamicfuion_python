import numpy as np


def compute_distance_matrix(point_set1: np.ndarray, point_set2: np.ndarray) -> np.ndarray:
    """
    Compute P1 x P2 distance matrix where P1 is the number of points in set 1 and P2 is the number of points in set 2.
    :param point_set1:
    :param point_set2:
    :return:
    """
    nodes_tiled = np.tile(point_set2, (len(point_set1), 1, 1))
    points_tiled = np.tile(point_set1, (1, len(point_set2))).reshape(len(point_set1), -1, 3)
    distance_matrix = np.linalg.norm(points_tiled - nodes_tiled, axis=2)
    return distance_matrix
