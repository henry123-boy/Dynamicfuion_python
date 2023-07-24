#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 7/6/23.
#  Copyright (c) 2023 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import sys
from typing import List, Tuple

import numpy as np


def compute_block_sparse_product_count(
        a: List[Tuple[int, int, np.ndarray]],
        n_a: int,
        m_a: int,
        b: List[Tuple[int, int, np.ndarray]],
        n_b: int,
        m_b: int
) -> int:
    a_coordinate_set = set([(i, j) for (i, j, block) in a])
    b_coordinate_set = set([(i, j) for (i, j, block) in b])
    assert m_a == n_b
    count = 0
    for i_a in range(0, n_a):
        for j_a in range(0, m_a):
            if (i_a, j_a) in a_coordinate_set:
                for j_b in range(0, m_b):
                    if (j_a, j_b) in b_coordinate_set:
                        count += 1
    return count


def index_block(matrix: np.ndarray, block_size: int, i: int, j: int) -> np.ndarray:
    return matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]


def main():
    a_blocks = [(0, 0, np.array([[3., 4.],
                                 [5., 6.]])),
                (0, 1, np.array([[1., 2.],
                                 [3., 4.]])),
                (1, 2, np.array([[5., 6.],
                                 [7., 8.]]))]
    n_a = 2
    m_a = 3
    b_blocks = [(0, 0, np.array([[1., 2.],
                                 [3., 4.]])),
                (1, 1, np.array([[5., 6.],
                                 [7., 8.]])),
                (2, 0, np.array([[9., 10.],
                                 [11., 0.]]))]
    n_b = 3
    m_b = 2

    block_product_count = compute_block_sparse_product_count(a_blocks, n_a, m_a, b_blocks, n_b, m_b)
    print(f"Block product count: {block_product_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
