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


def indexed_blocks_to_block_lookup_structure(indexed_block_list: List[Tuple[int, int, np.ndarray]], min_i: int = 0,
                                             min_j: int = 0):
    block_dict_2d = {}
    for (i, j, block) in indexed_block_list:
        if i > min_i and j > min_j:
            if i not in block_dict_2d:
                block_dict_2d[i] = {}
            block_dict_2d[i][j] = block
    return block_dict_2d


def indexed_zero_block_list_like(block_list: List[Tuple[int, int, np.ndarray]]) -> List[Tuple[int, int, np.ndarray]]:
    return [(i, j, np.zeros_like(block)) for (i, j, block) in block_list]


def indexed_zero_block_list(coordinate_list: List[Tuple[int, int]], block_size: int) -> List[
    Tuple[int, int, np.ndarray]]:
    return [(i, j, np.zeros((block_size, block_size))) for (i, j, block) in coordinate_list]


def generate_diagonal_coordinates(start: int, end: int):
    return [(i, i) for i in range(start, end, 1)]


def blocks_as_diag_coordinate_indexed(blocks: List[np.ndarray], start: int, end: int):
    assert len(blocks) == end - start
    return [(i, i, blocks[i]) for i in range(start, end, 1)]


def cholesky_upper_triangular_from_sparse_H(hessian_blocks_diagonal: List[np.ndarray],
                                            hessian_blocks_upper: List[Tuple[int, int, np.ndarray]],
                                            hessian_blocks_upper_corner: List[Tuple[int, int, np.ndarray]],
                                            layer_node_counts: np.ndarray) -> np.ndarray:
    first_layer_node_count = layer_node_counts[0]
    L_diag_upper_left = [scipy.linalg.cholesky(block, lower=True) for block in
                         hessian_blocks_diagonal[:first_layer_node_count]]
    L_inv_diag_upper_left = [np.linalg.inv(L) for L in L_diag_upper_left]

    U_diag_upper_left = [L.T for L in L_diag_upper_left]
    U_upper = [(i, j, L_inv_diag_upper_left[i].dot(U_h)) for (i, j, U_h) in hessian_blocks_upper]
    block_size = hessian_blocks_diagonal[0].shape[0]
    node_count = len(hessian_blocks_diagonal)
    U = np.zeros((block_size * node_count, block_size * node_count))

    fill_in_diagonal_blocks(U, U_diag_upper_left)
    fill_sparse_blocks(U, U_upper, flip_i_and_j=False)

    U_other_layer_blocks = U_upper
    H_other_layer_blocks = hessian_blocks_upper + hessian_blocks_upper_corner \
                           + blocks_as_diag_coordinate_indexed(hessian_blocks_diagonal[:first_layer_node_count],
                                                               first_layer_node_count, node_count)

    U_other_layer_dict = indexed_blocks_to_block_lookup_structure(U_upper)
    H_other_layer_dict = indexed_blocks_to_block_lookup_structure(H_other_layer_blocks)



    return U


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


if __name__ == "__main__":
    sys.exit(main())
