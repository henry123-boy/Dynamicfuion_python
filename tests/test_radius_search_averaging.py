import numpy as np
import math
import timeit

from warp_field.radius_search_averaging import downsample_radius_search_averaging
from warp_field.distance_matrix import compute_distance_matrix


def test_pythonic_radius_search_averaging():
    point_count = 1000
    seed = int(np.random.rand() * 100)
    np.random.seed(seed)
    points = (np.random.random((point_count, 3)) - 0.5) * 100.0
    search_radius = 10
    benchmark_iteration_count = 5
    benchmark = False
    start_time = timeit.default_timer()
    for i in range(benchmark_iteration_count if benchmark else 1):
        downsampled = downsample_radius_search_averaging(points, search_radius)
    if benchmark:
        print()
        print(f"Execution time (ave over {benchmark_iteration_count} runs): ",
              (timeit.default_timer() - start_time) / benchmark_iteration_count)
    distance_matrix = compute_distance_matrix(downsampled, downsampled)

    distances = distance_matrix.flatten()
    distances = distances[np.nonzero(distances)]

    assert (distances.min() > search_radius)
