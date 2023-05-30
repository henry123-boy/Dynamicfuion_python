#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 5/26/23.
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
import math
import sys
from typing import List, Tuple, Dict

import numpy as np
import scipy
import scipy.linalg
import cv2


def extract_diagonal_blocks(matrix: np.ndarray, block_size: int) -> List[np.ndarray]:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert block_size > 0
    assert matrix.shape[0] % block_size == 0
    diagonal_blocks = []
    for i_block in range(0, matrix.shape[0] // block_size):
        i_start = i_block * block_size
        i_end = (i_block + 1) * block_size
        diagonal_blocks.append(matrix[i_start:i_end, i_start:i_end])

    return diagonal_blocks


def fill_in_diagonal_blocks(matrix, blocks: List[np.ndarray]):
    assert len(blocks) > 0
    block0 = blocks[0]
    assert len(block0.shape) == 2
    assert block0.shape[0] == block0.shape[1]
    block_size = block0.shape[0]
    for i_block, block in enumerate(blocks):
        i_start = i_block * block_size
        i_end = (i_block + 1) * block_size
        matrix[i_start:i_end, i_start:i_end] = block


def matrix_from_diagonal_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    block0 = blocks[0]
    block_count = len(blocks)
    block_size = block0.shape[0]
    matrix = np.zeros((block_count * block_size, block_count * block_size))
    fill_in_diagonal_blocks(matrix, blocks)
    return matrix


def precondition_diagonal_blocks(blocks: List[np.ndarray], factor: float):
    for block in blocks:
        block += np.eye(block.shape[0]) * factor


def compare_block_lists(blocks0: List[np.ndarray], blocks1: List[np.ndarray]) -> np.ndarray:
    return np.array([np.allclose(block0, block1) for block0, block1 in zip(blocks0, blocks1)])


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def unfold_edge_jacobian(condensed):
    dEi_dR = skew(condensed)
    dEi_dt = np.eye(3) * condensed[3]
    dEj_dt = np.eye(3) * condensed[4]
    return dEi_dR, dEi_dt, dEj_dt


def edge_jacobians_and_edges_to_full_jacobian(edges: np.ndarray, edge_jacobians: np.ndarray,
                                              node_count: int) -> np.ndarray:
    edge_count = len(edges)
    J = np.zeros((3 * edge_count, 6 * node_count))
    for i_edge, (J_edge_condensed, edge) in enumerate(zip(edge_jacobians, edges)):
        i = edge[0]
        j = edge[1]
        dEi_dR, dEi_dt, dEj_dt = unfold_edge_jacobian(J_edge_condensed)
        J[i_edge * 3: i_edge * 3 + 3, i * 6: i * 6 + 3] = dEi_dR
        J[i_edge * 3: i_edge * 3 + 3, i * 6 + 3: i * 6 + 6] = dEi_dt
        J[i_edge * 3: i_edge * 3 + 3, j * 6 + 3: j * 6 + 6] = dEj_dt
    return J


def index_block(matrix: np.ndarray, block_size: int, i: int, j: int) -> np.ndarray:
    return matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]


def set_block(matrix: np.ndarray, block_size: int, i: int, j: int, block: np.ndarray) -> None:
    matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block


def compute_sparse_H(edges: np.ndarray, edge_jacobians: np.ndarray,
                     node_count: int, layer_node_counts: np.ndarray) \
        -> Tuple[List[np.ndarray], List[Tuple[int, int, np.ndarray]], List[Tuple[int, int, np.ndarray]]]:
    node_jacobians = {}
    for i_edge, (J_edge_condensed, edge) in enumerate(zip(edge_jacobians, edges)):
        i = edge[0]
        j = edge[1]
        if i not in node_jacobians:
            node_jacobians[i] = []
        node_jacobians[i].append((J_edge_condensed, 'i'))
        if j not in node_jacobians:
            node_jacobians[j] = []
        node_jacobians[j].append((J_edge_condensed, 'j'))

    hessian_blocks_upper = []
    hessian_blocks_upper_corner = []
    first_layer_node_count = layer_node_counts[0]
    for i_edge, (J_edge_condensed, edge) in enumerate(zip(edge_jacobians, edges)):
        i = edge[0]
        j = edge[1]
        dEi_dR, dEi_dt, dEj_dt = unfold_edge_jacobian(J_edge_condensed)
        dEi = np.hstack((dEi_dR, dEi_dt))
        dEj = np.hstack((np.zeros((3, 3)), dEj_dt))
        H_ij = dEi.T.dot(dEj)
        if i >= first_layer_node_count and j >= first_layer_node_count:
            hessian_blocks_upper_corner.append((i, j, H_ij))
        else:
            hessian_blocks_upper.append((i, j, H_ij))

    hessian_blocks_diagonal = np.ndarray((node_count, 6, 6))
    for i_node, node_edge_jacobian_list in node_jacobians.items():
        J_virtual_column = np.ndarray((len(node_edge_jacobian_list) * 3, 6))
        for i_node_edge_jacobian, (J_edge_condensed, J_type) in enumerate(node_edge_jacobian_list):
            i_start_virtual_row = i_node_edge_jacobian * 3
            i_end_virtual_row = i_start_virtual_row + 3
            if J_type == 'i':
                dEi_dR = skew(J_edge_condensed)
                dEi_dt = np.eye(3) * J_edge_condensed[3]
                dEi = np.hstack((dEi_dR, dEi_dt))
                J_virtual_column[i_start_virtual_row:i_end_virtual_row] = dEi
            else:  # J_type == 'j'
                dEj_dt = np.eye(3) * J_edge_condensed[4]
                dEj = np.hstack((np.zeros((3, 3)), dEj_dt))
                J_virtual_column[i_start_virtual_row:i_end_virtual_row] = dEj
        hessian_blocks_diagonal[i_node] = (J_virtual_column.T.dot(J_virtual_column))
    return list(hessian_blocks_diagonal), hessian_blocks_upper, hessian_blocks_upper_corner


def fill_sparse_blocks(matrix: np.array, sparse_blocks: List[Tuple[int, int, np.ndarray]], flip_i_and_j: bool = False,
                       transpose: bool = False):
    block_size = sparse_blocks[0][2].shape[0]

    def fill_block(matrix, i, j):
        i_start = i * block_size
        i_end = i_start + block_size
        j_start = j * block_size
        j_end = j_start + block_size
        matrix[i_start:i_end, j_start:j_end] = block.T if transpose else block

    if flip_i_and_j:
        for j, i, block in sparse_blocks:
            fill_block(matrix, i, j)
    else:
        for i, j, block in sparse_blocks:
            fill_block(matrix, i, j)


def sparse_H_to_dense(hessian_blocks_diagonal: List[np.ndarray],
                      hessian_blocks_upper: List[Tuple[int, int, np.ndarray]]) -> np.ndarray:
    diagonal_block_count = len(hessian_blocks_diagonal)
    block_size = hessian_blocks_diagonal[0].shape[0]
    H = np.zeros((block_size * diagonal_block_count, block_size * diagonal_block_count))
    fill_in_diagonal_blocks(H, hessian_blocks_diagonal)

    fill_sparse_blocks(H, hessian_blocks_upper, flip_i_and_j=False)
    fill_sparse_blocks(H, hessian_blocks_upper, flip_i_and_j=True, transpose=True)
    return H


def cholesky_upper_triangular_from_sparse_H(hessian_blocks_diagonal: List[np.ndarray],
                                            hessian_blocks_upper: List[Tuple[int, int, np.ndarray]],
                                            hessian_blocks_upper_corner: List[Tuple[int, int, np.ndarray]],
                                            layer_node_counts: np.ndarray) -> np.ndarray:
    L_diag = [scipy.linalg.cholesky(block, lower=True) for block in hessian_blocks_diagonal]
    L_inv_diag = [np.linalg.inv(L) for L in L_diag]
    U_diag = [L.T for L in L_diag]
    U_upper = [(i, j, L_inv_diag[i].dot(U_h)) for (i, j, U_h) in hessian_blocks_upper]
    block_size = hessian_blocks_diagonal[0].shape[0]
    diagonal_block_count = len(hessian_blocks_diagonal)
    U = np.zeros((block_size * diagonal_block_count, block_size * diagonal_block_count))

    fill_in_diagonal_blocks(U, U_diag)
    fill_sparse_blocks(U, U_upper, flip_i_and_j=False)

    first_layer_node_count = layer_node_counts[0]

    corner_blocks_diagonal = hessian_blocks_diagonal[first_layer_node_count:]
    corner_blocks_upper = [(i - first_layer_node_count, j - first_layer_node_count, block) for (i, j, block) in
                           hessian_blocks_upper_corner]
    H_corner = sparse_H_to_dense(corner_blocks_diagonal, corner_blocks_upper)
    # for i, j, block in U_upper:
    #     if j > first_layer_node_count:
    #         corner_i = i - first_layer_node_count
    #         corner_j = j - first_layer_node_count
    #         print(i, j, corner_i, corner_j)
    #         H_corner[corner_i * 6:(corner_i + 1) * 6, corner_j * 6: (corner_j + 1) * 6] -= block.T.dot(block)
    #         H_corner[corner_j * 6: (corner_j + 1) * 6, corner_i * 6:(corner_i + 1) * 6] -= block.dot(block.T)

    U[first_layer_node_count * 6:, first_layer_node_count * 6:] = scipy.linalg.cholesky(H_corner, lower=False)
    return U


def cholesky_banachiewicz(A: np.ndarray) -> np.ndarray:
    matrix_row_count = A.shape[0]
    L = np.zeros_like(A)
    for i in range(0, matrix_row_count):
        for j in range(0, i+1):
            sum = 0.0
            for k in range(0, j):
                sum += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = math.sqrt(A[i, i] - sum)
            else:
                L[i, j] = 1.0 / L[j, j] * (A[i, j] - sum)
    return L

# def cholesky_banachiewicz_block(A: np.ndarray, block_size) -> np.ndarray:
#     matrix_row_count = A.shape[0]
#     L = np.zeros_like(A)
#
#     for i in range(0,)


def main():
    np.set_printoptions(suppress=True)
    J = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/J.npy")
    H_gt = J.T.dot(J)
    edges = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/edges.npy")
    edge_jacobians = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/edge_jacobians.npy")
    layer_node_counts = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/layer_node_counts.npy")
    node_count = 249
    H_diag, H_upper, H_upper_corner = compute_sparse_H(edges, edge_jacobians, node_count, layer_node_counts)
    H = sparse_H_to_dense(H_diag, H_upper + H_upper_corner)
    print("H reconstructed successfully: ", np.allclose(H_gt, H))
    lm_factor = 0.001
    precondition_diagonal_blocks(H_diag, lm_factor)
    U = cholesky_upper_triangular_from_sparse_H(H_diag, H_upper, H_upper_corner, layer_node_counts)
    H_aug = H + np.eye(H.shape[0]) * lm_factor
    U_gt = scipy.linalg.cholesky(H_aug, lower=False)
    print("H factorized successfully: ", np.allclose(U, U_gt))
    U_comparison_im = (np.isclose(U, U_gt)).astype(np.uint8) * 255
    cv2.imwrite('/mnt/Data/Reconstruction/output/matrix_experiments/U_comparison_im.png', U_comparison_im)

    return 0


def main1():
    # @formatter:off
    A = np.array([[1.87713618, 1.13720606, 0.        , 0.        , 0.43085978, 1.17323464],
                  [1.13720606, 0.77520692, 0.        , 0.        , 0.20166133, 0.6677495 ],
                  [0.        , 0.        , 0.29459216, 0.52935895, 0.        , 0.        ],
                  [0.        , 0.        , 0.52935895, 0.97976523, 0.        , 0.        ],
                  [0.43085978, 0.20166133, 0.        , 0.        , 0.48812625, 0.77752449],
                  [1.17323464, 0.6677495 , 0.        , 0.        , 0.77752449, 1.60339873]])
    # @formatter:on
    L = cholesky_banachiewicz(A)
    print(L)

    return 0


if __name__ == "__main__":
    sys.exit(main1())
