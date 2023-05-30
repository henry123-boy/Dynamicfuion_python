#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 5/30/23.
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
from typing import Dict, List, Tuple

import numpy as np


def generate_random_block_matrix(size: int, block_size: int, block_density=0.5) \
        -> np.ndarray:
    assert size > 0 and block_size > 0 and 0.0 <= block_density <= 1.0
    assert size % block_size == 0
    matrix = np.zeros((size, size))
    size_blocks = size // block_size
    block_count = size_blocks * size_blocks
    block_indices = np.arange(0, block_count)
    filled_blocks = set(np.random.choice(block_indices, int(block_density * block_count), False))
    for i in range(0, size_blocks):
        for j in range(0, size_blocks):
            block_index = i * size_blocks + j
            if block_index in filled_blocks:
                block = np.random.rand(block_size, block_size)
                matrix[i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size] = block
    return matrix


def generate_random_symmetric_block_matrix(size: int, block_size: int, block_density=0.5) \
        -> np.ndarray:
    assert size > 0 and block_size > 0 and 0.0 <= block_density <= 1.0
    assert size % block_size == 0
    matrix = np.zeros((size, size))
    size_blocks = size // block_size
    lower_block_indices = np.tril_indices(size_blocks)
    block_count = len(lower_block_indices[0])
    block_indices = np.arange(0, block_count)
    filled_blocks = set(np.random.choice(block_indices, int(block_density * block_count), False))
    for i_block_linear, (i, j) in enumerate(zip(lower_block_indices[0], lower_block_indices[1])):
        if i_block_linear in filled_blocks:
            if i == j:
                block = np.random.rand(block_size, block_size)
                block = (block + block.T) / 2
                matrix[i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size] = block
            else:
                block = np.random.rand(block_size, block_size)
                matrix[i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size] = block
                matrix[j * block_size: (j + 1) * block_size,
                i * block_size: (i + 1) * block_size] = block.T
    return matrix


def generate_random_block_matrix_and_block_rows(size: int, block_size: int, block_density=0.5) \
        -> Tuple[np.ndarray, Dict[int, List[Tuple[int, np.ndarray]]]]:
    assert size > 0 and block_size > 0 and 0.0 <= block_density <= 1.0
    assert size % block_size == 0
    matrix = np.zeros((size, size))
    size_blocks = size // block_size
    block_count = size_blocks * size_blocks
    block_indices = np.arange(0, block_count)
    filled_blocks = set(np.random.choice(block_indices, int(block_density * block_count), False))
    block_rows = {}
    for i in range(0, size_blocks):
        for j in range(0, size_blocks):
            block_index = i * size_blocks + j
            if block_index in filled_blocks:
                if i not in block_rows:
                    block_rows[i] = []
                block = np.random.rand(block_size, block_size)
                matrix[i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size] = block
                block_rows[i].append((j, block))
    return matrix, block_rows


def extract_blocks(matrix: np.ndarray, block_size: int) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    assert matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]
    assert block_size > 0
    assert size % block_size == 0
    size_blocks = size // block_size
    block_rows = {}
    for i in range(0, size_blocks):
        for j in range(0, size_blocks):
            block = matrix[i * block_size: (i + 1) * block_size,
                    j * block_size: (j + 1) * block_size]
            if np.count_nonzero(block) > 0:
                if i not in block_rows:
                    block_rows[i] = []
                block_rows[i].append((j, block))
    return block_rows


def precondition_matrix(A: np.ndarray, lm_factor=0.001) -> np.ndarray:
    assert A.shape[0] == A.shape[1]
    return A + np.eye(A.shape[0]) * lm_factor


def generate_random_block_positive_semidefinite(size: int, block_size: int, block_density=0.5, lm_factor=0.001):
    B = generate_random_block_matrix(size, block_size, block_density)
    A = B.T @ B
    return precondition_matrix(A, lm_factor)
