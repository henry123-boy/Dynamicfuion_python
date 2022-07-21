#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph) on 10/27/21.
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
import gc
from typing import Tuple, Union, List
from timeit import default_timer as timer
import math
import numpy as np
import torch

import kornia

if kornia.__version__ >= '0.5.0':
    import kornia.geometry.conversions as kornia

from alignment.common.linear_solver_lu import LinearSolverLU
from settings import DeformNetParameters


class PointCloudAlignmentOptimizer:
    def __init__(self, output_gn_point_clouds=False):
        self.output_gn_point_clouds = output_gn_point_clouds

        self.gn_debug = DeformNetParameters.gn_debug.value

        self.gn_print_timings = DeformNetParameters.gn_print_timings.value

        self.gn_use_edge_weighting = DeformNetParameters.gn_use_edge_weighting.value
        self.gn_check_condition_num = DeformNetParameters.gn_check_condition_num.value
        self.gn_break_on_condition_num = DeformNetParameters.gn_break_on_condition_num.value
        self.gn_max_condition_num = DeformNetParameters.gn_max_condition_num.value

        self.gn_num_iter = DeformNetParameters.gn_num_iter.value
        self.gn_data_flow = DeformNetParameters.gn_data_flow.value
        self.gn_data_depth = DeformNetParameters.gn_data_depth.value
        self.gn_arap = DeformNetParameters.gn_arap.value
        self.gn_lm_factor = DeformNetParameters.gn_lm_factor.value

        self.lambda_data_flow = math.sqrt(self.gn_data_flow)
        self.lambda_data_depth = math.sqrt(self.gn_data_depth)
        self.lambda_arap = math.sqrt(self.gn_arap)

        vec_to_skew_mat_np = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        self.vec_to_skew_mat = torch.from_numpy(vec_to_skew_mat_np).to('cuda')

    def optimize_nodes(self, match_count: int, optimized_node_count: int, batch_edge_count: int,
                       graph_nodes_i: torch.Tensor,
                       source_anchors: torch.Tensor, source_weights: torch.Tensor,
                       source_points_filtered: torch.Tensor, source_colors_filtered: torch.Tensor,
                       correspondence_weights_filtered: torch.Tensor,
                       xy_pixels_warped_filtered: torch.Tensor,
                       target_matches_filtered: torch.Tensor,
                       graph_edge_pairs_filtered: torch.Tensor, graph_edge_weights_pairs: torch.Tensor,
                       num_neighbors: int,
                       fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                       batch_convergence_info) -> Tuple[
        bool, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]
    ]:

        self.vec_to_skew_mat.to(source_anchors.device)

        float_dtype = source_weights.dtype
        device = source_weights.device
        gauss_newton_iteration_count = self.gn_num_iter

        # The parameters in GN solver are 3 parameters for rotation and 3 parameters for
        # translation for every node. All node rotation parameters are listed first, and
        # then all node translation parameters are listed.
        #                        transform_delta = [rotations_current, translations_current]
        rotations_current = torch.eye(3, dtype=float_dtype, device=device).view(1, 3, 3).repeat(optimized_node_count, 1,
                                                                                                1)
        translations_current = torch.zeros((optimized_node_count, 3, 1), dtype=float_dtype, device=device)

        if self.gn_debug:
            print(
                f"\tMatch count: {match_count} || Node count: {optimized_node_count} || Edges count: {batch_edge_count}")

        # Initialize helper structures.
        data_increment_vec_0_3 = torch.arange(0, match_count * 3, 3, out=torch.cuda.LongTensor(),
                                              device=device)  # (match_count)
        data_increment_vec_1_3 = torch.arange(1, match_count * 3, 3, out=torch.cuda.LongTensor(),
                                              device=device)  # (match_count)
        data_increment_vec_2_3 = torch.arange(2, match_count * 3, 3, out=torch.cuda.LongTensor(),
                                              device=device)  # (match_count)

        arap_increment_vec_0_3 = None
        arap_increment_vec_1_3 = None
        arap_increment_vec_2_3 = None
        arap_one_vec = None
        if batch_edge_count > 0:
            arap_increment_vec_0_3 = torch.arange(0, batch_edge_count * 3, 3, out=torch.cuda.LongTensor(),
                                                  device=device)  # (batch_edge_count)
            arap_increment_vec_1_3 = torch.arange(1, batch_edge_count * 3, 3, out=torch.cuda.LongTensor(),
                                                  device=device)  # (batch_edge_count)
            arap_increment_vec_2_3 = torch.arange(2, batch_edge_count * 3, 3, out=torch.cuda.LongTensor(),
                                                  device=device)  # (batch_edge_count)
            arap_one_vec = torch.ones(batch_edge_count, dtype=float_dtype, device=device)

        ill_posed_system = False
        residuals = None

        gn_point_clouds = []
        for i_iteration in range(gauss_newton_iteration_count):
            residual_data, jacobian_data = \
                self.compute_data_residual_and_jacobian(data_increment_vec_0_3, data_increment_vec_1_3,
                                                        data_increment_vec_2_3,
                                                        match_count, optimized_node_count, graph_nodes_i,
                                                        source_anchors, source_weights, source_points_filtered,
                                                        source_colors_filtered,
                                                        correspondence_weights_filtered, xy_pixels_warped_filtered,
                                                        target_matches_filtered,
                                                        i_iteration, fx, fy, cx, cy, rotations_current,
                                                        translations_current, gn_point_clouds)
            loss_arap = None
            if batch_edge_count > 0:
                residual_arap, jacobian_arap = \
                    self.compute_arap_residual_and_jacobian(arap_increment_vec_0_3, arap_increment_vec_1_3,
                                                            arap_increment_vec_2_3, arap_one_vec,
                                                            graph_edge_pairs_filtered, graph_edge_weights_pairs,
                                                            graph_nodes_i, batch_edge_count, optimized_node_count,
                                                            num_neighbors,
                                                            rotations_current, translations_current)
                loss_arap = torch.norm(residual_arap).item()
                residuals = torch.cat((residual_data, residual_arap), 0)
                jacobian = torch.cat((jacobian_data, jacobian_arap), 0)
            else:
                residuals = residual_data
                jacobian = jacobian_data

            success, transform_delta = self.solve_linear_system(residuals, jacobian, batch_convergence_info)
            ill_posed_system = not success
            if ill_posed_system:
                break

            # Increment the current rotation and translation.
            rotation_increments = kornia.angle_axis_to_rotation_matrix(
                transform_delta[:optimized_node_count * 3].view(optimized_node_count, 3))
            translation_increments = transform_delta[optimized_node_count * 3:].view(optimized_node_count, 3, 1)

            rotations_current = torch.matmul(rotation_increments, rotations_current)
            translations_current = translations_current + translation_increments

            loss_data = torch.norm(residual_data).item()
            loss_total = torch.norm(residuals).item()

            batch_convergence_info["data"].append(loss_data)
            batch_convergence_info["total"].append(loss_total)

            if batch_edge_count > 0:
                batch_convergence_info["arap"].append(loss_arap)

            if self.gn_debug:
                if batch_edge_count > 0:
                    print(f"\t\t-->Iteration: {i_iteration}. "
                          f"Loss: \tdata = {loss_data:.3f}, \tarap = {loss_arap:.3f}, \ttotal = {loss_total:.3f}")
                else:
                    print(
                        f"\t\t-->Iteration: {i_iteration}. Loss: \tdata = {loss_data:.3f}, \ttotal = {loss_total:.3f}")
        return ill_posed_system, residuals, rotations_current, translations_current, gn_point_clouds

    def solve_linear_system(self,
                            residual, jacobian,
                            batch_convergence_info: dict) -> Tuple[bool, Union[None, torch.Tensor]]:
        lm_factor = self.gn_lm_factor

        timer_system_start = timer()

        # Compute A = (J^T)J and b = -(J^T)r.
        jacobian_transpose = torch.transpose(jacobian, 0, 1)
        A = torch.matmul(jacobian_transpose, jacobian)
        b = torch.matmul(-jacobian_transpose, residual)

        # Solve linear system Ax = b.
        A = A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * lm_factor

        assert torch.isfinite(A).all(), A

        if self.gn_print_timings:
            print("\t\tSystem computation: {:.3f} s".format(timer() - timer_system_start))
        timer_cond_start = timer()

        # Check the determinant/condition number.
        # If unstable, we break optimization.
        if self.gn_check_condition_num:
            with torch.no_grad():
                # Condition number.
                values = torch.linalg.eigvals(A)
                real_values = torch.view_as_real(values)[:, 0]
                assert torch.isfinite(real_values).all(), real_values
                max_eig_value = torch.max(torch.abs(real_values))
                min_eig_value = torch.min(torch.abs(real_values))
                condition_number = max_eig_value / min_eig_value
                condition_number = condition_number.item()
                batch_convergence_info["condition_numbers"].append(condition_number)

                if self.gn_break_on_condition_num and (
                        not math.isfinite(condition_number) or condition_number > self.gn_max_condition_num):
                    print("\t\tToo high condition number: {0:e} (max: {1:.3f}, min: {2:.3f}). Discarding sample".format(
                        condition_number,
                        max_eig_value.item(),
                        min_eig_value.item()))
                    batch_convergence_info["errors"].append(
                        "Too high condition number: {0:e} (max: {1:.3f}, min: {2:.3f}). Discarding sample".format(
                            condition_number,
                            max_eig_value.item(),
                            min_eig_value.item()))
                    return False, None
                elif self.gn_debug:
                    print("\t\tCondition number: {0:e} (max: {1:.3f}, min: {2:.3f})".format(condition_number,
                                                                                            max_eig_value.item(),
                                                                                            min_eig_value.item()))

        if self.gn_print_timings:
            print("\t\tComputation of cond. num.: {:.3f} s".format(timer() - timer_cond_start))
        timer_solve_start = timer()

        linear_solver = LinearSolverLU.apply

        try:
            # transform_delta is x in Ax = b.
            transform_delta = linear_solver(A, b)
        except RuntimeError as e:
            print("\t\tSolver failed: Ill-posed system!", e)
            batch_convergence_info["errors"].append("Solver failed: ill-posed system!")
            return False, None

        if not torch.isfinite(transform_delta).all():
            print("\t\tSolver failed: Non-finite solution x!")
            batch_convergence_info["errors"].append("Solver failed: non-finite solution x!")
            return False, None

        if self.gn_print_timings:
            print("\t\tLinear solve: {:.3f} s".format(timer() - timer_solve_start))
        return True, transform_delta

    def compute_data_residual_and_jacobian(self,
                                           data_increment_vec_0_3: torch.Tensor,
                                           data_increment_vec_1_3: torch.Tensor,
                                           data_increment_vec_2_3: torch.Tensor,
                                           match_count: int,
                                           opt_num_nodes_i: int,
                                           graph_nodes_i: torch.Tensor,
                                           source_anchors: torch.Tensor,
                                           source_weights: torch.Tensor,
                                           source_points_filtered: torch.Tensor,
                                           source_colors_filtered: torch.Tensor,
                                           correspondence_weights_filtered: torch.Tensor,
                                           xy_pixels_warped_filtered: torch.Tensor,
                                           target_matches_filtered: torch.Tensor,
                                           i_iteration: int,
                                           # camera projection parameters
                                           fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                                           rotations_current: torch.Tensor,
                                           translations_current: torch.Tensor,
                                           gn_point_clouds: List[Tuple[torch.Tensor, torch.Tensor]]
                                           ) -> Tuple[torch.Tensor, torch.Tensor]:

        timer_data_start = timer()
        jacobian_data = torch.zeros((match_count * 3, opt_num_nodes_i * 6), dtype=rotations_current.dtype,
                                    device=rotations_current.device)  # (num_matches*3, opt_num_nodes_i*6)
        deformed_points = torch.zeros((match_count, 3, 1), dtype=rotations_current.dtype,
                                      device=rotations_current.device)

        for k in range(4):  # Our data uses 4 anchors for every point
            node_idxs_k = source_anchors[:, k]  # (num_matches)
            nodes_k = graph_nodes_i[node_idxs_k].view(match_count, 3, 1)  # (num_matches, 3, 1)

            # Compute deformed point contribution.
            # (num_matches, 3, 1) = (num_matches, 3, 3) * (num_matches, 3, 1)
            rotated_points_k = torch.matmul(rotations_current[node_idxs_k],
                                            source_points_filtered - nodes_k)
            deformed_points_k = rotated_points_k + nodes_k + translations_current[node_idxs_k]
            deformed_points += \
                source_weights[:, k].view(match_count, 1, 1).repeat(1, 3, 1) * \
                deformed_points_k  # (num_matches, 3, 1)

        if self.output_gn_point_clouds:
            gn_point_clouds.append((deformed_points, source_colors_filtered))

        # Get necessary components of deformed points.
        eps = 1e-7  # Just as good practice, although matches should all have valid depth at this stage

        deformed_x = deformed_points[:, 0, :].view(match_count)  # (num_matches)
        deformed_y = deformed_points[:, 1, :].view(match_count)  # (num_matches)
        deformed_z_inverse = torch.div(1.0, deformed_points[:, 2, :].view(match_count) + eps)  # (num_matches)
        fx_mul_x = fx * deformed_x  # (num_matches)
        fy_mul_y = fy * deformed_y  # (num_matches)
        fx_div_z = fx * deformed_z_inverse  # (num_matches)
        fy_div_z = fy * deformed_z_inverse  # (num_matches)
        fx_mul_x_div_z = fx_mul_x * deformed_z_inverse  # (num_matches)
        fy_mul_y_div_z = fy_mul_y * deformed_z_inverse  # (num_matches)
        minus_fx_mul_x_div_z_2 = -fx_mul_x_div_z * deformed_z_inverse  # (num_matches)
        minus_fy_mul_y_div_z_2 = -fy_mul_y_div_z * deformed_z_inverse  # (num_matches)

        lambda_data_flow = self.lambda_data_flow
        lambda_data_depth = self.lambda_data_depth

        for k in range(4):  # Our data uses 4 anchors for every point
            node_idxs_k = source_anchors[:, k]  # (num_matches)
            nodes_k = graph_nodes_i[node_idxs_k].view(match_count, 3, 1)  # (num_matches, 3, 1)

            weights_k = source_weights[:, k] * correspondence_weights_filtered  # (num_matches)

            # Compute skew-symmetric part.
            rotated_points_k = torch.matmul(rotations_current[node_idxs_k],
                                            source_points_filtered - nodes_k)  # (num_matches, 3, 1) = (num_matches, 3, 3) * (num_matches, 3, 1)
            weighted_rotated_points_k = weights_k.view(match_count, 1, 1).repeat(1, 3,
                                                                                 1) * rotated_points_k  # (num_matches, 3, 1)
            skew_symetric_mat_data = -torch.matmul(self.vec_to_skew_mat, weighted_rotated_points_k).view(match_count, 3,
                                                                                                         3)  # (num_matches, 3, 3)

            # Compute jacobian wrt. TRANSLATION.
            # FLOW PART
            jacobian_data[data_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_k + 0] += \
                lambda_data_flow * weights_k * fx_div_z  # (num_matches)
            jacobian_data[data_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_k + 2] += \
                lambda_data_flow * weights_k * minus_fx_mul_x_div_z_2  # (num_matches)
            jacobian_data[data_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_k + 1] += \
                lambda_data_flow * weights_k * fy_div_z  # (num_matches)
            jacobian_data[data_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_k + 2] += \
                lambda_data_flow * weights_k * minus_fy_mul_y_div_z_2  # (num_matches)

            # DEPTH PART
            jacobian_data[data_increment_vec_2_3, 3 * opt_num_nodes_i + 3 * node_idxs_k + 2] += \
                lambda_data_depth * weights_k  # (num_matches)

            # Compute jacobian wrt. ROTATION.
            # FLOW PART
            jacobian_data[data_increment_vec_0_3, 3 * node_idxs_k + 0] += \
                lambda_data_flow * fx_div_z * skew_symetric_mat_data[:, 0, 0] + \
                minus_fx_mul_x_div_z_2 * skew_symetric_mat_data[:, 2, 0]

            jacobian_data[data_increment_vec_0_3, 3 * node_idxs_k + 1] += \
                lambda_data_flow * fx_div_z * skew_symetric_mat_data[:, 0, 1] + \
                minus_fx_mul_x_div_z_2 * skew_symetric_mat_data[:, 2, 1]

            jacobian_data[data_increment_vec_0_3, 3 * node_idxs_k + 2] += \
                lambda_data_flow * fx_div_z * skew_symetric_mat_data[:, 0, 2] + \
                minus_fx_mul_x_div_z_2 * skew_symetric_mat_data[:, 2, 2]

            jacobian_data[data_increment_vec_1_3, 3 * node_idxs_k + 0] += \
                lambda_data_flow * fy_div_z * skew_symetric_mat_data[:, 1, 0] + \
                minus_fy_mul_y_div_z_2 * skew_symetric_mat_data[:, 2, 0]

            jacobian_data[data_increment_vec_1_3, 3 * node_idxs_k + 1] += \
                lambda_data_flow * fy_div_z * skew_symetric_mat_data[:, 1, 1] + \
                minus_fy_mul_y_div_z_2 * skew_symetric_mat_data[:, 2, 1]

            jacobian_data[data_increment_vec_1_3, 3 * node_idxs_k + 2] += \
                lambda_data_flow * fy_div_z * skew_symetric_mat_data[:, 1, 2] + \
                minus_fy_mul_y_div_z_2 * skew_symetric_mat_data[:, 2, 2]

            # DEPTH PART
            jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 0] += \
                lambda_data_depth * skew_symetric_mat_data[:, 2, 0]
            jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 1] += \
                lambda_data_depth * skew_symetric_mat_data[:, 2, 1]
            jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 2] += \
                lambda_data_depth * skew_symetric_mat_data[:, 2, 2]

            assert torch.isfinite(jacobian_data).all(), jacobian_data

        residuals_data = torch.zeros((match_count * 3, 1), dtype=rotations_current.dtype,
                                     device=rotations_current.device)

        # FLOW PART
        # [projected deformed point u-coordinate] - [flow-warped source u-coordinate (u + Δu)]
        residuals_data[data_increment_vec_0_3, 0] = \
            lambda_data_flow * correspondence_weights_filtered * (
                    fx_mul_x_div_z + cx - xy_pixels_warped_filtered[:, 0, :].view(match_count))
        # [projected deformed point v-coordinate] - [flow-warped source v-coordinate (v + Δv)]
        residuals_data[data_increment_vec_1_3, 0] = \
            lambda_data_flow * correspondence_weights_filtered * (
                    fy_mul_y_div_z + cy - xy_pixels_warped_filtered[:, 1, :].view(match_count))

        # DEPTH PART
        # [projected deformed point u-coordinate] - [target matched point z-coordinate]
        residuals_data[data_increment_vec_2_3, 0] = \
            lambda_data_depth * correspondence_weights_filtered * (
                    deformed_points[:, 2, :] - target_matches_filtered[:, 2, :]).view(match_count)
        if self.gn_print_timings:
            print("\t\tData term: {:.3f} s".format(timer() - timer_data_start))
        return residuals_data, jacobian_data

    def compute_arap_residual_and_jacobian(self,
                                           arap_increment_vec_0_3: torch.Tensor,
                                           arap_increment_vec_1_3: torch.Tensor,
                                           arap_increment_vec_2_3: torch.Tensor,
                                           arap_one_vec: torch.Tensor,
                                           graph_edge_pairs_filtered: torch.Tensor,
                                           graph_edge_weights_pairs: torch.Tensor,
                                           graph_nodes_i: torch.Tensor,
                                           batch_edge_count: int,
                                           opt_num_nodes_i: int,
                                           num_neighbors: int,
                                           rotations_current: torch.Tensor,
                                           translations_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        timer_arap_start = timer()

        # TODO: initialize somewhere else and zero out here instead to avoid extra memory allocations + deallocations
        jacobian_arap = torch.zeros((batch_edge_count * 3, opt_num_nodes_i * 6), dtype=arap_one_vec.dtype,
                                    device=arap_one_vec.device)  # (batch_edge_count*3, opt_num_nodes_i*6)

        node_idxs_0 = graph_edge_pairs_filtered[:, 0]  # i node
        node_idxs_1 = graph_edge_pairs_filtered[:, 1]  # j node

        w = torch.ones_like(graph_edge_weights_pairs)
        if self.gn_use_edge_weighting:
            # Since graph edge weights sum up to 1 for all neighbors, we multiply
            # it by the number of neighbors to make the setting in the same scale
            # as in the case of not using edge weights (they are all 1 then).
            w = float(num_neighbors) * graph_edge_weights_pairs

        w_repeat = w.unsqueeze(-1).repeat(1, 3).unsqueeze(-1)
        w_repeat_repeat = w_repeat.repeat(1, 1, 3)

        nodes_0 = graph_nodes_i[node_idxs_0].view(batch_edge_count, 3, 1)
        nodes_1 = graph_nodes_i[node_idxs_1].view(batch_edge_count, 3, 1)

        lambda_arap = self.lambda_arap

        # Compute residual.
        # Rotated edges (between nodes)
        rotated_node_delta = torch.matmul(rotations_current[node_idxs_0], nodes_1 - nodes_0)  # (batch_edge_count, 3)
        # node 0 + node 0 translation + rotated edge to node 1 = node 0 current position + rotated edge to node 1
        #                        vs.
        # node 1 + node 1 translation = node 1 current position
        residuals_arap = lambda_arap * w_repeat * (rotated_node_delta + nodes_0 + translations_current[node_idxs_0]
                                                   - (nodes_1 + translations_current[node_idxs_1]))
        residuals_arap = residuals_arap.view(batch_edge_count * 3, 1)

        # Compute jacobian wrt. translations.
        jacobian_arap[arap_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 0] += \
            lambda_arap * w * arap_one_vec  # (batch_edge_count)
        jacobian_arap[arap_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 1] += \
            lambda_arap * w * arap_one_vec  # (batch_edge_count)
        jacobian_arap[arap_increment_vec_2_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 2] += \
            lambda_arap * w * arap_one_vec  # (batch_edge_count)

        jacobian_arap[arap_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 0] += \
            -lambda_arap * w * arap_one_vec  # (batch_edge_count)
        jacobian_arap[arap_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 1] += \
            -lambda_arap * w * arap_one_vec  # (batch_edge_count)
        jacobian_arap[arap_increment_vec_2_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 2] += \
            -lambda_arap * w * arap_one_vec  # (batch_edge_count)

        # Compute Jacobian wrt. rotations.
        # Derivative wrt. R_1 is equal to 0.
        skew_symmetric_mat_arap = \
            -lambda_arap * w_repeat_repeat * \
            torch.matmul(self.vec_to_skew_mat, rotated_node_delta).view(
                batch_edge_count, 3, 3
            )  # (batch_edge_count, 3, 3)

        jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 0] += skew_symmetric_mat_arap[:, 0, 0]
        jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 1] += skew_symmetric_mat_arap[:, 0, 1]
        jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 2] += skew_symmetric_mat_arap[:, 0, 2]
        jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 0] += skew_symmetric_mat_arap[:, 1, 0]
        jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 1] += skew_symmetric_mat_arap[:, 1, 1]
        jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 2] += skew_symmetric_mat_arap[:, 1, 2]
        jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 0] += skew_symmetric_mat_arap[:, 2, 0]
        jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 1] += skew_symmetric_mat_arap[:, 2, 1]
        jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 2] += skew_symmetric_mat_arap[:, 2, 2]

        assert torch.isfinite(jacobian_arap).all(), jacobian_arap

        if self.gn_print_timings:
            print("\t\tARAP term: {:.3f} s".format(timer() - timer_arap_start))
        return residuals_arap, jacobian_arap
