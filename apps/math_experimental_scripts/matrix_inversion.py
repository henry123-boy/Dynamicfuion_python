#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 6/5/23.
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
from typing import List

import numpy as np
from enum import Enum

PROGRAM_EXIT_SUCCESS = 0


class UpLo(Enum):
    UPPER = 0
    LOWER = 1


def multi_block_trtri(matrices: List[np.ndarray], uplo: UpLo) -> List[np.ndarray]:
    assert len(matrices) > 0
    first_matrix = matrices[0]
    assert len(first_matrix.shape) == 2
    assert first_matrix.shape[0] == first_matrix.shape[1]
    assert first_matrix.dtype == np.float32 or first_matrix.dtype == np.float64
    N = first_matrix.shape[0]
    matrices_even_count = matrices
    padded = False
    if len(matrices) % 2 == 1:
        matrices_even_count = matrices + [np.eye(N)]
        padded = True

    inverted = []

    for i_matrix_pair in range(len(matrices) // 2):
        i_matrix_a = i_matrix_pair * 2
        i_matrix_b = i_matrix_pair * 2 + 1
        matrix_a = matrices_even_count[i_matrix_a]
        matrix_b = matrices_even_count[i_matrix_b]

        inv_matrix_a = np.zeros_like(matrix_a)
        inv_matrix_b = np.zeros_like(matrix_a)

        if uplo == UpLo.UPPER:
            for i_col_a in range(N - 1, -1, -1):
                i_col_b = N - i_col_a - 1
                inv_matrix_a[i_col_a, i_col_a] = 1.0 / matrix_a[i_col_a, i_col_a]
                inv_matrix_b[i_col_b, i_col_b] = 1.0 / matrix_b[i_col_b, i_col_b]
                for i_row in range(i_col_a - 1, -1, -1):
                    matrix_diag_entry = matrix_a[i_row, i_row]
                    sum = 0
                    for k in range(i_row + 1, i_col_a + 1, 1):
                        sum += matrix_a[i_row, k] * inv_matrix_a[k, i_col_a]
                    inv_matrix_a[i_row, i_col_a] = -sum / matrix_diag_entry
                for i_row in range(i_col_b - 1, -1, -1):
                    matrix_diag_entry = matrix_b[i_row, i_row]
                    sum = 0
                    for k in range(i_row + 1, i_col_b + 1, 1):
                        sum += matrix_b[i_row, k] * inv_matrix_b[k, i_col_b]
                    inv_matrix_b[i_row, i_col_b] = -sum / matrix_diag_entry
        else:
            for i_col_a in range(0, N, 1):
                i_col_b = N - i_col_a - 1
                inv_matrix_a[i_col_a, i_col_a] = 1.0 / matrix_a[i_col_a, i_col_a]
                inv_matrix_b[i_col_b, i_col_b] = 1.0 / matrix_b[i_col_b, i_col_b]
                for i_row in range(i_col_a + 1, N, 1):
                    matrix_diag_entry = matrix_a[i_row, i_row]
                    sum = 0
                    for k in range(i_col_a, i_row, 1):
                        sum += matrix_a[i_row, k] * inv_matrix_a[k, i_col_a]
                    inv_matrix_a[i_row, i_col_a] = -sum / matrix_diag_entry
                for i_row in range(i_col_b + 1, N, 1):
                    matrix_diag_entry = matrix_b[i_row, i_row]
                    sum = 0
                    for k in range(i_col_b, i_row, 1):
                        sum += matrix_b[i_row, k] * inv_matrix_b[k, i_col_b]
                    inv_matrix_b[i_row, i_col_b] = -sum / matrix_diag_entry
        inverted.append(inv_matrix_a)
        inverted.append(inv_matrix_b)

    if padded:
        inverted = inverted[:-1]

    return inverted


def trtri(matrix: np.ndarray, uplo: UpLo) -> np.ndarray:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == np.float32 or matrix.dtype == np.float64
    N = matrix.shape[0]
    inv_matrix = np.zeros_like(matrix)
    if uplo == UpLo.UPPER:
        for i_col in range(N - 1, -1, -1):
            inv_matrix[i_col, i_col] = 1.0 / matrix[i_col, i_col]
            for i_row in range(i_col - 1, -1, -1):
                matrix_diag_entry = matrix[i_row, i_row]
                sum = 0
                for k in range(i_row + 1, i_col + 1, 1):
                    sum += matrix[i_row, k] * inv_matrix[k, i_col]
                inv_matrix[i_row, i_col] = -sum / matrix_diag_entry
    else:
        for i_col in range(0, N, 1):
            inv_matrix[i_col, i_col] = 1.0 / matrix[i_col, i_col]
            for i_row in range(i_col + 1, N, 1):
                matrix_diag_entry = matrix[i_row, i_row]
                sum = 0
                for k in range(i_col, i_row, 1):
                    sum += matrix[i_row, k] * inv_matrix[k, i_col]
                inv_matrix[i_row, i_col] = -sum / matrix_diag_entry
    return inv_matrix


def main() -> int:
    np.set_printoptions(suppress=True, edgeitems=40, linewidth=140)
    print("matrix U:")
    U1 = np.array([[4, 5, 6],
                   [0, 2, 3],
                   [0, 0, 1]], np.float32)
    print(U1)
    U1_inverse = trtri(U1, UpLo.UPPER)
    print("inverse: ")
    print(U1_inverse)
    print("U1 @ inverse")
    print(U1 @ U1_inverse)

    L = np.array([[1, 0, 0],
                  [2, 3, 0],
                  [4, 5, 6]], np.float32)
    print("matrix L:")
    print(L)
    inverse = trtri(L, UpLo.LOWER)
    print("inverse:")
    print(inverse)
    print("L @ inverse:")
    print(L @ inverse)

    N = 10
    U_large_flat_a = np.arange(len(np.triu_indices(N)[0]), 0, -1)
    U_large_a = np.zeros((N, N), dtype=np.float32)
    U_large_a[np.triu_indices(N)] = U_large_flat_a
    inverse = trtri(U_large_a, UpLo.UPPER)
    print(f"Inverse of {N} x {N} matrix A successful: {np.allclose(U_large_a @ inverse, np.eye(N), 1, 1e-6)} ")

    U_large_flat_b = np.arange(len(np.triu_indices(N)[0]) + len(np.triu_indices(N)[0]), len(np.triu_indices(N)[0]), -1)
    U_large_b = np.zeros((N, N), dtype=np.float32)
    U_large_b[np.triu_indices(N)] = U_large_flat_b
    inverse = trtri(U_large_b, UpLo.UPPER)
    print(f"Inverse of {N} x {N} matrix B successful: {np.allclose(U_large_b @ inverse, np.eye(N), 1, 1e-6)} ")

    U2 = np.array([[10, 11, 12],
                   [0, 8, 9],
                   [0, 0, 7]], np.float32)
    U2_inverse = trtri(U2, UpLo.UPPER)
    U_matrices_12 = [U1, U2]
    print (U_matrices_12)
    U_inverted_12 = multi_block_trtri(U_matrices_12, UpLo.UPPER)

    print(f"Batch inverse of {3} x {3} matrix A successful:"
          f" {np.allclose(U1 @ U_inverted_12[0], np.eye(3), 1, 1e-6)} ")
    print(f"Batch inverse of {3} x {3} matrix B successful:"
          f" {np.allclose(U2 @ U_inverted_12[1], np.eye(3), 1, 1e-6)} ")


    U_matrices_AB = [U_large_a, U_large_b]

    U_inverted_AB = multi_block_trtri(U_matrices_AB, UpLo.UPPER)
    print(f"Batch inverse of {N} x {N} matrix A upper successful:"
          f" {np.allclose(U_large_a @ U_inverted_AB[0], np.eye(N), 1, 1e-6)} ")
    print(f"Batch inverse of {N} x {N} matrix B upper successful:"
          f" {np.allclose(U_large_b @ U_inverted_AB[1], np.eye(N), 1, 1e-6)} ")
    
    L_matrices_AB = [U_large_a.T, U_large_b.T]
    L_inverted_AB = multi_block_trtri(L_matrices_AB, UpLo.LOWER)
    print(f"Batch inverse of {N} x {N} matrix A lower successful:"
          f" {np.allclose(L_matrices_AB[0] @ L_inverted_AB[0], np.eye(N), 1, 1e-6)} ")
    print(f"Batch inverse of {N} x {N} matrix B lower successful:"
          f" {np.allclose(L_matrices_AB[1] @ L_inverted_AB[1], np.eye(N), 1, 1e-6)} ")

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
