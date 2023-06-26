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
from pathlib import Path
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
    # associate jacobians with their node / block column
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
                       transpose: bool = False, block_offsets: Tuple[int, int] = (0, 0)):
    block_size = sparse_blocks[0][2].shape[0]

    def fill_block(matrix, i, j):
        i_start = (i - block_offsets[0]) * block_size
        i_end = i_start + block_size
        j_start = (j - block_offsets[1]) * block_size
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


def sparse_U_to_dense(U_blocks_diagonal: List[np.ndarray],
                      U_blocks_upper: List[Tuple[int, int, np.ndarray]]) -> np.ndarray:
    diagonal_block_count = len(U_blocks_diagonal)
    block_size = U_blocks_diagonal[0].shape[0]
    U = np.zeros((block_size * diagonal_block_count, block_size * diagonal_block_count))
    fill_in_diagonal_blocks(U, U_blocks_diagonal)
    fill_sparse_blocks(U, U_blocks_upper, flip_i_and_j=False)
    return U


def indexed_blocks_to_block_lookup_structure(indexed_block_list: List[Tuple[int, int, np.ndarray]], min_i: int = 0,
                                             min_j: int = 0) -> Dict[Tuple[int, int], np.ndarray]:
    block_dict_2d = {}
    for (i, j, block) in indexed_block_list:
        if i >= min_i and j >= min_j:
            block_dict_2d[(i, j)] = block
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


# fill in the remaining blocks in the corner for U, i.e. the cholesky decomposition of H, based on the existing entries
# in sparse H dict & U dict
def cholesky_blocked_sparse_corner(U_block_dict: Dict[Tuple[int, int], np.ndarray],
                                   H_block_dict: Dict[Tuple[int, int], np.ndarray],
                                   H_corner_diagonal_blocks: List[np.ndarray],
                                   corner_offset: int, node_count: int, block_size: int) \
        -> Tuple[List[np.ndarray], List[Tuple[int, int, np.ndarray]]]:
    assert len(H_corner_diagonal_blocks) == node_count - corner_offset

    corner_U_diagonal_blocks = []
    corner_U_upper_blocks = []

    i_diagonal = 0

    #__DEBUG
    product_count = 0
    #__DEBUG
    inspected_row = 209

    # i -- index of current block row in output
    for i in range(corner_offset, node_count):
        #__DEBUG
        block_row = []
        row_product_count = 0
        block_sum = np.zeros((block_size, block_size), dtype=np.float64)
        # use block column at index i to augment the matrix diagonal entry
        for k in range(0, i):
            if (k, i) in U_block_dict:
                U_ki = U_block_dict[(k, i)]
                block_sum += U_ki.transpose() @ U_ki
                row_product_count += 1
                if i == 210:
                    print(f"Level {i} above-diagonal-block sum product for [{k}, {i}]: ")
                    print(U_ki)
                    print(U_ki.transpose() @ U_ki)
                #__DEBUG
                # if i == inspected_row:
                #     print(f"[{k},{i},{k},{i}],")

        #__DEBUG
        if i == 210:
            print(f"Level {i} above-diagonal-block sum: ")
            print(block_sum)
        # Update U-matrix diagonal blocks
        H_ii = H_corner_diagonal_blocks[i_diagonal]
        U_ii = scipy.linalg.cholesky(H_ii - block_sum, lower=False)
        corner_U_diagonal_blocks.append(U_ii)
        L_kk_inv = np.linalg.inv(U_ii.T)
        #__DEBUG
        block_row.append(U_ii)

        # Update U-matrix blocks above the diagonal
        # j is the index of block column in output
        for j in range(i + 1, node_count):
            block_sum = np.zeros((block_size, block_size), dtype=np.float64)

            for k in range(0, i):  # k is the row index again here, we traverse all rows before i
                if (k, i) in U_block_dict and (k, j) in U_block_dict:
                    U_ki = U_block_dict[(k, i)]
                    U_kj = U_block_dict[(k, j)]
                    block_sum += U_ki.transpose() @ U_kj
                    row_product_count += 1
                    #__DEBUG
                    # if i == inspected_row:
                    #     print(f"[{k},{i},{k},{j}],")

            # update "inner" matrix blocks
            if (i, j) in H_block_dict:
                H_ij = H_block_dict[(i, j)]
            else:
                H_ij = np.zeros((block_size, block_size), dtype=np.float64)

            if i == 210:
                print(f"Level {i} above-{j}-block sum: ")
                print(block_sum)
            H_ij_new = H_ij - block_sum

            #__DEBUG
            # if i == inspected_row and j == 244:
            #     print(f"H_ij for i=={inspected_row}, j==244:")
            #     print(H_ij)
            #     print(f"block_sum for i=={inspected_row}, j==244:")
            #     print(block_sum)
            #     print(f"H_ij_new for i=={inspected_row}, j==244:")
            #     print(H_ij_new)
            #     print(f"L_kk_inv for i=={inspected_row}, j==244:")
            #     print(L_kk_inv)

            U_ij = L_kk_inv @ H_ij_new
            U_block_dict[(i, j)] = U_ij
            corner_U_upper_blocks.append((i, j, U_ij))
            #__DEBUG
            block_row.append(U_ij)
        i_diagonal += 1
        #__DEBUG
        # if i == inspected_row:
        #     print(f"Block row for i={i}:")
        #     print(np.array(block_row))
        print(f"Product count during \"arrowhead\" corner factorization for level {i}: {row_product_count}")
        product_count += row_product_count

    #__DEBUG
    print(f"Total product count during \"arrowhead\" corner factorization: {product_count}")

    return corner_U_diagonal_blocks, corner_U_upper_blocks


def cholesky_upper_triangular_from_sparse_H(hessian_blocks_diagonal: List[np.ndarray],
                                            hessian_blocks_upper: List[Tuple[int, int, np.ndarray]],
                                            hessian_blocks_upper_corner: List[Tuple[int, int, np.ndarray]],
                                            layer_node_counts: np.ndarray, save_cpp_test_data: bool = False) \
        -> Tuple[List[np.ndarray], List[Tuple[int, int, np.ndarray]]]:
    first_layer_node_count = layer_node_counts[0]
    L_diag_upper_left = [scipy.linalg.cholesky(block, lower=True) for block in
                         hessian_blocks_diagonal[:first_layer_node_count]]
    L_inv_diag_upper_left = [np.linalg.inv(L) for L in L_diag_upper_left]

    U_diag_upper_left = [L.T for L in L_diag_upper_left]
    U_upper_right = [(i, j, L_inv_diag_upper_left[i] @ U_h) for (i, j, U_h) in hessian_blocks_upper]
    block_size = hessian_blocks_diagonal[0].shape[0]
    node_count = len(hessian_blocks_diagonal)

    U_block_dict = indexed_blocks_to_block_lookup_structure(U_upper_right)
    H_block_dict = indexed_blocks_to_block_lookup_structure(hessian_blocks_upper + hessian_blocks_upper_corner)

    U_diag_lower_right, U_lower_right = \
        cholesky_blocked_sparse_corner(U_block_dict, H_block_dict, hessian_blocks_diagonal[first_layer_node_count:],
                                       first_layer_node_count, node_count, block_size)

    if save_cpp_test_data:
        corner_size_blocks = len(hessian_blocks_diagonal) - len(U_diag_upper_left)
        np.save("/mnt/Data/Reconstruction/output/matrix_experiments/U_diag_upper_left.npy",
                np.array(U_diag_upper_left))
        U_upper = [L_inv_diag_upper_left[i] @ U_h if i < first_layer_node_count else np.zeros((block_size, block_size))
                   for (i, j, U_h) in hessian_blocks_upper + hessian_blocks_upper_corner]
        np.save("/mnt/Data/Reconstruction/output/matrix_experiments/U_upper.npy",
                np.array(U_upper))
        U_lower_right_dense = np.zeros((corner_size_blocks * block_size, corner_size_blocks * block_size),
                                       dtype=np.float32)
        fill_sparse_blocks(U_lower_right_dense, U_lower_right,
                           block_offsets=(first_layer_node_count, first_layer_node_count))
        fill_in_diagonal_blocks(U_lower_right_dense, U_diag_lower_right)
        np.save("/mnt/Data/Reconstruction/output/matrix_experiments/U_lower_right_dense.npy", U_lower_right_dense)

    return U_diag_upper_left + U_diag_lower_right, U_upper_right + U_lower_right


def build_sparse_lookup_structure(indexed_blocks: List[Tuple[int, int, np.ndarray]], transpose: bool = False):
    dict = {}
    if transpose:
        # column lookup
        for (i, j, block) in indexed_blocks:
            if j not in dict:
                dict[j] = []
            dict[j].append((i, block.T))
        # __DEBUG
        max = 0
        for key, value in dict.items():
            if len(value) > max:
                max = len(value)
        print(f"Tallest column in blocks: {max}")
    else:
        # row lookup
        for (i, j, block) in indexed_blocks:
            if i not in dict:
                dict[i] = []
            dict[i].append((j, block))
        # __DEBUG
        max = 0
        for key, value in dict.items():
            if len(value) > max:
                max = len(value)
        print(f"Longest row in blocks: {max}")

    return dict


def solve_triangular_sparse_back_substitution(u_blocks_diagonal: List[np.ndarray],
                                              u_blocks_upper: List[Tuple[int, int, np.ndarray]],
                                              y: np.ndarray):
    block_row_count = len(u_blocks_diagonal)
    block_size = u_blocks_diagonal[0].shape[0]
    assert y.shape == (block_row_count * block_size,)
    solution = np.ndarray((block_row_count * block_size))
    row_dict = build_sparse_lookup_structure(u_blocks_upper, transpose=False)
    for i_block_row in range(block_row_count - 1, -1, -1):
        row_u_diag = u_blocks_diagonal[i_block_row]
        i_block_start = i_block_row * block_size
        i_block_end = (i_block_row + 1) * block_size
        row_y_prime = y[i_block_start:i_block_end].copy()
        if i_block_row in row_dict:
            for (j, block) in row_dict[i_block_row]:
                j_block_start = j * block_size
                j_block_end = (j + 1) * block_size
                row_j_x = solution[j_block_start:j_block_end]
                row_y_prime -= block @ row_j_x

        row_i_x = scipy.linalg.solve_triangular(row_u_diag, row_y_prime, lower=False)
        solution[i_block_start:i_block_end] = row_i_x
    return solution


def solve_triangular_sparse_forward_substitution(u_blocks_diagonal: List[np.ndarray],
                                                 u_blocks_upper: List[Tuple[int, int, np.ndarray]],
                                                 b: np.ndarray):
    block_column_count = len(u_blocks_diagonal)
    block_size = u_blocks_diagonal[0].shape[0]
    assert b.shape == (block_column_count * block_size,)
    solution = np.ndarray((block_column_count * block_size))
    column_dict = build_sparse_lookup_structure(u_blocks_upper, transpose=True)
    for j_block_column in range(0, block_column_count, 1):
        column_l_diag = u_blocks_diagonal[j_block_column].T
        i_block_start = j_block_column * block_size
        i_block_end = (j_block_column + 1) * block_size
        column_b_prime = b[i_block_start:i_block_end].copy()
        if j_block_column in column_dict:
            for (i, block) in column_dict[j_block_column]:
                j_block_start = i * block_size
                j_block_end = (i + 1) * block_size
                column_j_x = solution[j_block_start:j_block_end]
                column_b_prime -= block @ column_j_x

        row_i_y = scipy.linalg.solve_triangular(column_l_diag, column_b_prime, lower=True)
        solution[i_block_start:i_block_end] = row_i_y
    return solution


BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE = 80
BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE = 160


def generate_cpp_test_block_sparse_arrowhead_input_data(
        diagonal_blocks: List[np.array],
        upper_blocks: List[Tuple[int, int, np.ndarray]],
        layer_node_counts: np.ndarray,
        base_path: str = "/mnt/Data/Reconstruction/output/matrix_experiments"
):
    first_layer_node_count = layer_node_counts[0]
    node_count = len(diagonal_blocks)
    breadboard_width_nodes = node_count - first_layer_node_count
    breadboard = np.zeros((node_count, breadboard_width_nodes), np.int16)
    breadboard -= 1
    upper_column_block_lists = \
        np.zeros((breadboard_width_nodes, BLOCK_ARROWHEAD_COLUMN_BLOCK_MAX_COUNT_ESTIMATE, 2), np.int32)
    upper_column_block_counts = np.zeros((breadboard_width_nodes,), np.int32)
    upper_row_block_lists = \
        np.zeros((node_count, BLOCK_ARROWHEAD_ROW_BLOCK_MAX_COUNT_ESTIMATE, 2), np.int32)
    upper_row_block_counts = np.zeros((node_count,), np.int32)

    upper_block_list = []
    coordinate_list = []

    for i_block, (i, j, block) in enumerate(upper_blocks):
        j_breadboard = j - first_layer_node_count
        breadboard[i, j_breadboard] = i_block

        i_block_in_column = upper_column_block_counts[j_breadboard]
        upper_column_block_counts[j_breadboard] += 1
        upper_column_block_lists[j_breadboard, i_block_in_column, 0] = i
        upper_column_block_lists[j_breadboard, i_block_in_column, 1] = i_block

        j_block_in_row = upper_row_block_counts[i]
        upper_row_block_counts[i] += 1
        upper_row_block_lists[i, j_block_in_row, 0] = j
        upper_row_block_lists[i, j_block_in_row, 1] = i_block
        upper_block_list.append(block)
        coordinate_list.append((i, j))

    np.save(str(Path(base_path) / "diagonal_blocks.npy"), np.array(diagonal_blocks))
    np.save(str(Path(base_path) / "upper_blocks.npy"), np.array(upper_block_list))
    np.save(str(Path(base_path) / "upper_block_coordinates.npy"), np.array(coordinate_list))
    np.save(str(Path(base_path) / "breadboard.npy"), breadboard)
    np.save(str(Path(base_path) / "upper_column_block_lists.npy"), upper_column_block_lists)
    np.save(str(Path(base_path) / "upper_column_block_counts.npy"), upper_column_block_counts)
    np.save(str(Path(base_path) / "upper_row_block_lists.npy"), upper_row_block_lists)
    np.save(str(Path(base_path) / "upper_row_block_counts.npy"), upper_row_block_counts)


def main():
    np.set_printoptions(suppress=True, linewidth=350, edgeitems=100)
    J = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/J.npy")
    H_gt = J.T.dot(J)
    edges = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/edges.npy")
    edge_jacobians = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/edge_jacobians.npy")
    layer_node_counts = np.load("/mnt/Data/Reconstruction/output/matrix_experiments/layer_node_counts.npy")
    node_count = 249
    print(f"Node count: {node_count}")
    print(f"Count of blocks at arrowhead base: {layer_node_counts[0]}")
    print(f"Edge count: {len(edges)}")
    H_diag, H_upper, H_upper_corner = compute_sparse_H(edges, edge_jacobians, node_count, layer_node_counts)
    print(
        f"H total block count: {len(H_diag) + len(H_upper) + len(H_upper_corner)}, H diagonal block count: {len(H_diag)}, H upper block count: {len(H_upper) + len(H_upper_corner)}")
    H = sparse_H_to_dense(H_diag, H_upper + H_upper_corner)
    print("H computed as block-sparse from edges and reconstructed from sparse representation successfully: ",
          np.allclose(H_gt, H))
    lm_factor = 0.001
    precondition_diagonal_blocks(H_diag, lm_factor)
    save_cpp_test_data = False
    if save_cpp_test_data:
        generate_cpp_test_block_sparse_arrowhead_input_data(H_diag, H_upper + H_upper_corner, layer_node_counts)
    U_diag, U_upper = cholesky_upper_triangular_from_sparse_H(H_diag, H_upper, H_upper_corner, layer_node_counts,
                                                              save_cpp_test_data=save_cpp_test_data)
    U = sparse_U_to_dense(U_diag, U_upper)
    H_aug = H + np.eye(H.shape[0]) * lm_factor
    U_gt = scipy.linalg.cholesky(H_aug, lower=False)

    print("Block-sparse H factorized successfully: ", np.allclose(U, U_gt))

    dummy_negJr = np.random.rand(node_count * 6)

    y_gt = scipy.linalg.solve_triangular(U_gt.T, dummy_negJr, lower=True)
    y = solve_triangular_sparse_forward_substitution(U_diag, U_upper, dummy_negJr)
    print("Uy=b block-sparse forward-substitution finished successfully: ", np.allclose(y_gt, y))

    delta = solve_triangular_sparse_back_substitution(U_diag, U_upper, y)
    delta_gt = scipy.linalg.solve(H_aug, dummy_negJr)
    print("Lx=y block-sparse back-substitution finished successfully: ", np.allclose(delta, delta_gt))

    return 0


if __name__ == "__main__":
    sys.exit(main())
