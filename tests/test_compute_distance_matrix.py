import numpy as np
import math

from pipeline.numpy_cpu.distance_matrix import compute_distance_matrix


def test_compute_distance_matrix():
    x = np.array([
        [0, 0, 0],  # 0
        [0, 0, 1],  # 1
        [0, 1, 0],  # 2
        [0, 1, 1],  # 3
        [1, 0, 0],  # 4
        [1, 0, 1],  # 5
        [1, 1, 0],  # 6
        [1, 1, 1]  # 7
    ], dtype=np.float32)

    matrix_xx = compute_distance_matrix(x, x)
    sqrt_2 = math.sqrt(2.0)
    sqrt_3 = math.sqrt(3.0)

    matrix_xx_gt = np.array([
        [0, 1, 1, sqrt_2, 1, sqrt_2, sqrt_2, sqrt_3],
        [1, 0, sqrt_2, 1, sqrt_2, 1, sqrt_3, sqrt_2],
        [1, sqrt_2, 0, 1, sqrt_2, sqrt_3, 1, sqrt_2],
        [sqrt_2, 1, 1, 0, sqrt_3, sqrt_2, sqrt_2, 1],
        [1, sqrt_2, sqrt_2, sqrt_3, 0, 1, 1, sqrt_2],
        [sqrt_2, 1, sqrt_3, sqrt_2, 1, 0, sqrt_2, 1],
        [sqrt_2, sqrt_3, 1, sqrt_2, 1, sqrt_2, 0, 1],
        [sqrt_3, sqrt_2, sqrt_2, 1, sqrt_2, 1, 1, 0]
    ], dtype=np.float32)

    assert np.alltrue(matrix_xx == matrix_xx_gt)
