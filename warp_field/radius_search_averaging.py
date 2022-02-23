import numpy as np
from warp_field.distance_matrix import compute_distance_matrix


def downsample_radius_search_averaging_helper(points: np.ndarray, radius: float,
                                              distance_matrix: np.ndarray) -> np.ndarray:

    point_mask = np.ones(len(points), dtype=np.bool)
    downsampled = []
    for i_point, neighbor_distances in enumerate(distance_matrix):
        sorted_neighbors = np.argsort(neighbor_distances)
        if point_mask[i_point]:
            point_mask[i_point] = False
            cumulative_point = points[i_point]
            point_count = 1
            for i_neighbor in sorted_neighbors:
                if point_mask[i_neighbor]:
                    average_point = cumulative_point / point_count
                    neighbor = points[i_neighbor]
                    if np.linalg.norm(average_point - neighbor) > radius:
                        break
                    cumulative_point += neighbor
                    point_mask[i_neighbor] = False
                    point_count += 1
            average_point = cumulative_point / point_count
            downsampled.append(average_point)
    downsampled = np.array(downsampled)
    distance_matrix = compute_distance_matrix(downsampled, downsampled)
    distances = distance_matrix.flatten()
    distances = distances[np.nonzero(distances)]

    if distances.min() > radius or len(points) == 1:
        return downsampled
    else:
        return downsample_radius_search_averaging_helper(np.array(downsampled), radius, distance_matrix)


def downsample_radius_search_averaging(points: np.ndarray, radius: float) -> np.ndarray:
    distance_matrix = compute_distance_matrix(points, points)
    return downsample_radius_search_averaging_helper(points, radius, distance_matrix)
