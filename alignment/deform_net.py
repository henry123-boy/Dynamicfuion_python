from alignment import pwcnet

import torch
import torch.nn as nn
import numpy as np

from timeit import default_timer as timer
from typing import Tuple, Union, List

from alignment.point_cloud_alignment_optimizer import PointCloudAlignmentOptimizer
from settings import DeformNetParameters, AlignmentParameters
from alignment.mask_net import MaskNet


def make_pixel_coordinate_grid(image_height: int, image_width: int, batch_count: int,
                               device: torch.device) -> torch.Tensor:
    x_coords = torch.arange(image_width, dtype=torch.float32, device=device) \
        .unsqueeze(0).expand(image_height, image_width).unsqueeze(0)
    y_coords = torch.arange(image_height, dtype=torch.float32, device=device) \
        .unsqueeze(1).expand(image_height, image_width).unsqueeze(0)

    xy_coords = torch.cat([x_coords, y_coords], 0)
    xy_coords = xy_coords.unsqueeze(0).repeat(batch_count, 1, 1, 1)  # (bs, 2, 448, 640)
    return xy_coords


def project_using_correspondences(points: torch.Tensor, xy_pixels_warped: torch.Tensor, intrinsics: torch.Tensor,
                                  image_height: int, image_width: int) -> torch.Tensor:
    # ==== project points using correspondences ===
    # note: z-coordinate remains the same
    # x_proj = (z * u_warped - c_x) / f_x
    # y_proj = (z * v_warped - c_y) / f_y
    # z_proj = z
    points_z = points[:, 2, :, :].view(-1, 1, image_height, image_width)
    c_xy = intrinsics[:, 2:].view(-1, 2, 1, 1)
    f_xy = intrinsics[:, :2].view(-1, 2, 1, 1)
    return torch.cat((points_z * (xy_pixels_warped - c_xy) / f_xy, points_z), 1)


def flow_bidirectional_projection_error(points: torch.Tensor, pixel_flow_forward: torch.Tensor,
                                        pixel_flow_back: torch.Tensor, intrinsics: torch.Tensor,
                                        image_height: int, image_width: int, batch_count: int):
    z_source = points[:, 2, :, :].view(-1, 1, image_height, image_width)
    flow_forward_and_back = pixel_flow_forward + pixel_flow_back
    f_xy = intrinsics[:, :2].view(-1, 2, 1, 1)

    flow_camera_space = flow_forward_and_back * z_source / f_xy
    return torch.linalg.vector_norm(flow_camera_space, axis=1)


def compute_flow_pixel_and_normalized_targets(flow: torch.Tensor, image_height: int, image_width: int,
                                              batch_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ########################################################################
    # region Apply dense flow to warp the source points to target frame.
    ########################################################################
    xy_coords = make_pixel_coordinate_grid(image_height, image_width, batch_count, flow.device)

    # Apply the flow to pixel coordinates.
    xy_coords_warped = xy_coords + flow
    xy_pixels_warped = xy_coords_warped.clone()

    # Normalize the flow coordinates to be between -1 and 1 (i.e. camera-space).
    # Since we use "align_corners=False" during interpolation above, the boundaries of corner pixels
    # are -1 and 1, not their centers.
    xy_coords_warped[:, 0, :, :] = (xy_coords_warped[:, 0, :, :]) / (image_width - 1)
    xy_coords_warped[:, 1, :, :] = (xy_coords_warped[:, 1, :, :]) / (image_height - 1)
    xy_coords_warped = xy_coords_warped * 2 - 1

    # Permute the warped coordinates to fit the grid_sample format.
    xy_coords_warped = xy_coords_warped.permute(0, 2, 3, 1)
    return xy_pixels_warped, xy_coords_warped


class DeformNet(nn.Module):

    # TODO: to provide looser coupling, instead of passing in a TelemetryGenerator here, have a boolean parameter to
    #  optionally record the optimization-deformed point clouds at every step into the output dict from
    #  the forward method.
    def __init__(self, output_gn_point_clouds=False):
        super().__init__()

        self.skip_solver = DeformNetParameters.skip_solver.value

        self._depth_sampling_mode = DeformNetParameters.depth_sampling_mode.value.value
        self.gn_max_depth = DeformNetParameters.gn_max_depth.value
        self.gn_min_nodes = DeformNetParameters.gn_min_nodes.value
        self.gn_max_nodes = DeformNetParameters.gn_max_nodes.value
        self.gn_max_matches_train = DeformNetParameters.gn_max_matches_train.value
        self.gn_max_matches_train_per_batch = DeformNetParameters.gn_max_matches_train_per_batch.value
        self.gn_max_matches_eval = DeformNetParameters.gn_max_matches_eval.value
        self.gn_max_warped_points = DeformNetParameters.gn_max_warped_points.value
        self.gn_debug = DeformNetParameters.gn_debug.value

        self.gn_remove_clusters_with_few_matches = DeformNetParameters.gn_remove_clusters_with_few_matches.value
        self.gn_min_num_correspondences_per_cluster = DeformNetParameters.gn_min_num_correspondences_per_cluster.value

        self.alignment_image_width = AlignmentParameters.image_width.value
        self.alignment_image_height = AlignmentParameters.image_height.value

        self.use_mask = DeformNetParameters.use_mask.value
        self.threshold_mask_predictions = DeformNetParameters.threshold_mask_predictions.value
        self.threshold = DeformNetParameters.threshold.value
        self.patchwise_threshold_mask_predictions = DeformNetParameters.patchwise_threshold_mask_predictions.value
        self.patch_size = DeformNetParameters.patch_size.value

        self.optimizer = PointCloudAlignmentOptimizer(output_gn_point_clouds)

        # Optical flow network
        self.flow_net = pwcnet.PWCNet()
        if DeformNetParameters.freeze_optical_flow_net.value:
            # Freeze
            for param in self.flow_net.parameters():
                param.requires_grad = False

        # Weight prediction network
        self.mask_net = MaskNet(DeformNetParameters.use_batch_norm.value)
        if DeformNetParameters.freeze_mask_net.value:
            # Freeze
            for param in self.mask_net.parameters():
                param.requires_grad = False

    def forward(
            self,
            source: torch.Tensor, target: torch.Tensor,
            graph_nodes: torch.Tensor, graph_edges: torch.Tensor, graph_edges_weights: torch.Tensor,
            graph_clusters: torch.Tensor,
            pixel_anchors: torch.Tensor, pixel_weights: torch.Tensor, num_nodes_vec: torch.Tensor,
            intrinsics: torch.Tensor,
            evaluate: bool = False, split: str = "train"
    ):
        batch_count = source.shape[0]

        image_width = source.shape[3]
        image_height = source.shape[2]

        # We assume we always use 4 nearest anchors.
        assert pixel_anchors.shape[3] == 4

        convergence_info = []
        for i_batch in range(batch_count):
            convergence_info.append({
                "total": [],
                "arap": [],
                "data": [],
                "condition_numbers": [],
                "valid": 0,
                "errors": []
            })

        # clones avoid memory bugs here
        source_color = torch.clone(source[:, :3, :, :])
        target_color = torch.clone(target[:, :3, :, :])

        # features2: 2nd features layer of the encoder
        flow, flow2, flow3, flow4, flow5, flow6, features2 = \
            self.apply_flow_net(source_color, target_color, image_height, image_width)

        # xy_pixels_warped: 2D correspondence flow-field grid in raw, pixel coordinates,
        #   each vector represents the pixel coordinate + flow vector
        # xy_coords_warped: 2D correspondence flow-field grid in camera-space coordinates, i.e. -1 to 1 in
        #   both x & y axes; each vector represents the (normalized) pixel coordinate + flow vector
        xy_pixels_warped, xy_coords_warped = \
            compute_flow_pixel_and_normalized_targets(flow, image_height, image_width, batch_count)

        source_points = source[:, 3:, :, :].clone()
        source_colors = source[:, :3, :, :].clone()

        target_points = target[:, 3:, :, :].clone()  # get only 3D coordinates
        valid_source_points, target_matches, valid_target_matches, correspondence_mask, valid_correspondence_counts = \
            self.construct_point_to_point_correspondences(source_points, target_points, pixel_anchors, xy_coords_warped)

        mask_net_prediction = None
        # Initialize correspondence weights with 1's. We might overwrite them with MaskNet-predicted weights next.
        correspondence_weights = torch.ones(valid_target_matches.shape, dtype=torch.float32,
                                            device=valid_target_matches.device)

        # We predict correspondence weights [0, 1], if we use mask network.
        # We invalidate target matches later, when we assign a weight to each match.
        if self.use_mask:
            mask_net_prediction, correspondence_weights = \
                self.apply_mask_net(evaluate, target[:, :3, :, :].clone(), xy_coords_warped,
                                    source, target_matches, features2, batch_count, image_height, image_width)

        ################################################################################################################
        # Enforce bidirectional correspondence consistency
        ################################################################################################################
        if DeformNetParameters.enforce_bidirectional_consistency.value:
            flow_back, _, _, _, _, _, _ = self.apply_flow_net(target_color, source_color, image_height, image_width)
            projection_error = flow_bidirectional_projection_error(source_points, flow, flow_back, intrinsics,
                                                                   image_height, image_width, batch_count)
            bidirectional_correspondence_mask = \
                (projection_error < DeformNetParameters.bidirectional_consistency_threshold.value)
            correspondence_mask = correspondence_mask & bidirectional_correspondence_mask
            correspondence_weights = torch.where(bidirectional_correspondence_mask, correspondence_weights,
                                                 torch.zeros_like(correspondence_weights))

        ########################################################################
        # region Initialize graph structures.
        ########################################################################
        num_nodes_total = graph_nodes.shape[1]
        num_neighbors = graph_edges.shape[2]

        # We assume we always use 4 nearest anchors.
        assert pixel_anchors.shape[3] == 4

        node_rotations = torch.eye(3, dtype=source.dtype, device=source.device).view(1, 1, 3, 3).repeat(batch_count,
                                                                                                        num_nodes_total,
                                                                                                        1, 1)
        node_translations = torch.zeros((batch_count, num_nodes_total, 3), dtype=source.dtype, device=source.device)
        deformations_validity = torch.zeros((batch_count, num_nodes_total), dtype=source.dtype, device=source.device)
        valid_solve = torch.zeros(batch_count, dtype=torch.uint8, device=source.device)
        deformed_points_prediction = torch.zeros((batch_count, self.gn_max_warped_points, 3), dtype=source.dtype,
                                                 device=source.device)
        deformed_points_indices = torch.zeros((batch_count, self.gn_max_warped_points), dtype=torch.int64,
                                              device=source.device)
        deformed_points_subsampled = torch.zeros(batch_count, dtype=torch.uint8, device=source.device)

        # Optionally, skip the solver
        if not evaluate and self.skip_solver:
            return {
                "flow_data": [flow2, flow3, flow4, flow5, flow6],
                "node_rotations": node_rotations,
                "node_translations": node_translations,
                "deformations_validity": deformations_validity,
                "deformed_points_pred": deformed_points_prediction,
                "valid_solve": valid_solve,
                "mask_pred": mask_net_prediction,
                "correspondence_info": [
                    xy_coords_warped,
                    source_points, valid_source_points,
                    target_matches, valid_target_matches,
                    None, deformed_points_indices, deformed_points_subsampled
                ],
                "convergence_info": convergence_info,
                "weight_info": {
                    "total_corres_num": 0,
                    "total_corres_weight": 0.0
                }
            }
        # endregion
        ########################################################################
        # region Estimate node deformations using differentiable Gauss-Newton.
        ########################################################################
        correspondences_exist = valid_correspondence_counts > 0

        total_num_matches_per_batch = 0

        weight_info = {
            "total_corres_num": 0,
            "total_corres_weight": 0.0
        }

        gn_point_clouds = []

        for i_batch in range(batch_count):
            if self.gn_debug:
                print()
                print("--Sample", i_batch, "in batch--")

            num_nodes_i = num_nodes_vec[i_batch]
            if num_nodes_i > self.gn_max_nodes or num_nodes_i < self.gn_min_nodes:
                print("\tSolver failed: Invalid number of nodes: {0}".format(num_nodes_i))
                convergence_info[i_batch]["errors"].append(
                    "Solver failed: Invalid number of nodes: {0}".format(num_nodes_i))
                continue

            if not correspondences_exist[i_batch]:
                print("\tSolver failed: No valid correspondences before filtering")
                convergence_info[i_batch]["errors"].append("Solver failed: No valid correspondences before filtering")
                continue

            timer_start = timer()

            graph_nodes_i = graph_nodes[i_batch, :num_nodes_i, :]  # (num_nodes_i, 3)
            graph_edges_i = graph_edges[i_batch, :num_nodes_i, :]  # (num_nodes_i, 8)
            graph_edges_weights_i = graph_edges_weights[i_batch, :num_nodes_i, :]  # (num_nodes_i, 8)
            graph_clusters_i = graph_clusters[i_batch, :num_nodes_i, :]  # (num_nodes_i, 1)

            fx = intrinsics[i_batch, 0]
            fy = intrinsics[i_batch, 1]
            cx = intrinsics[i_batch, 2]
            cy = intrinsics[i_batch, 3]

            ###############################################################################################################
            # region Filter invalid matches.
            ###############################################################################################################
            valid_correspondences_idxs = torch.where(correspondence_mask[i_batch])

            source_points_filtered = source_points[i_batch].permute(1, 2, 0)
            source_points_filtered = source_points_filtered[valid_correspondences_idxs[0],
                                     valid_correspondences_idxs[1], :].view(-1, 3, 1)
            source_colors_filtered = source_colors[i_batch].permute(1, 2, 0)
            source_colors_filtered = source_colors_filtered[valid_correspondences_idxs[0],
                                     valid_correspondences_idxs[1], :].view(-1, 3, 1)

            target_matches_filtered = target_matches[i_batch].permute(1, 2, 0)
            target_matches_filtered = target_matches_filtered[valid_correspondences_idxs[0],
                                      valid_correspondences_idxs[1], :].view(-1, 3, 1)

            xy_pixels_warped_filtered = xy_pixels_warped[i_batch].permute(1, 2, 0)  # (height, width, 2)
            xy_pixels_warped_filtered = xy_pixels_warped_filtered[valid_correspondences_idxs[0],
                                        valid_correspondences_idxs[1], :].view(-1, 2, 1)

            correspondence_weights_filtered = correspondence_weights[
                i_batch, valid_correspondences_idxs[0], valid_correspondences_idxs[1]].view(
                -1)  # (match_count)
            # (match_count, 4)
            source_anchors = \
                pixel_anchors[i_batch, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]
            # (match_count, 4)
            source_weights = \
                pixel_weights[i_batch, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]

            match_count = source_points_filtered.shape[0]
            # endregion
            ############################################################################################################
            # region Generate weight info to estimate average weight.
            ############################################################################################################
            weight_info = {
                "total_corres_num": correspondence_weights_filtered.shape[0],
                "total_corres_weight": float(correspondence_weights_filtered.sum())
            }
            # endregion
            ############################################################################################################
            # region Randomly subsample matches, if necessary.
            ############################################################################################################
            if split == "val" or split == "test":
                max_num_matches = self.gn_max_matches_eval
            elif split == "train":
                max_num_matches = self.gn_max_matches_train
            else:
                raise Exception("Split {} is not defined".format(split))

            if match_count > max_num_matches:
                sampled_indexes = torch.randperm(match_count)[:max_num_matches]
                source_points_filtered = source_points_filtered[sampled_indexes]
                source_colors_filtered = source_colors_filtered[sampled_indexes]
                target_matches_filtered = target_matches_filtered[sampled_indexes]
                xy_pixels_warped_filtered = xy_pixels_warped_filtered[sampled_indexes]
                correspondence_weights_filtered = correspondence_weights_filtered[sampled_indexes]
                source_anchors = source_anchors[sampled_indexes]
                source_weights = source_weights[sampled_indexes]

                match_count = max_num_matches
            # endregion

            ############################################################################################################
            # region Remove nodes if their corresponding clusters don't have enough correspondences
            # (Correspondences that have "bad" nodes as anchors will also be removed)
            ############################################################################################################
            map_opt_nodes_to_complete_nodes_i = list(range(0, num_nodes_i))
            optimized_node_count = num_nodes_i

            if self.gn_remove_clusters_with_few_matches:
                source_anchors_numpy = source_anchors.clone().cpu().numpy()
                source_weights_numpy = source_weights.clone().cpu().numpy()

                # Compute number of correspondences (or matches) per node in the form of the
                # match weight sum
                match_weights_per_node = np.zeros(num_nodes_i)

                # This method adds weight contribution of each match to the corresponding node,
                # allowing duplicate node ids in the flattened array.
                np.add.at(match_weights_per_node, source_anchors_numpy.flatten(), source_weights_numpy.flatten())

                total_match_weights = 0.0
                match_weights_per_cluster = {}
                for node_id in range(num_nodes_i):
                    # Get sum of weights for current node.
                    match_weights = match_weights_per_node[node_id]

                    # Get cluster id for current node
                    cluster_id = graph_clusters_i[node_id].item()

                    if cluster_id in match_weights_per_cluster:
                        match_weights_per_cluster[cluster_id] += match_weights
                    else:
                        match_weights_per_cluster[cluster_id] = match_weights

                    total_match_weights += match_weights

                # we'll build a mask that stores which nodes will survive
                valid_nodes_mask_i = torch.ones((num_nodes_i), dtype=torch.bool, device=source.device)

                # if not enough matches in a cluster, mark all cluster's nodes for removal
                node_ids_for_removal = []
                for cluster_id, cluster_match_weights in match_weights_per_cluster.items():
                    if self.gn_debug:
                        print('cluster_id', cluster_id, cluster_match_weights)

                    if cluster_match_weights < self.gn_min_num_correspondences_per_cluster:
                        index_equals_cluster_id = torch.where(graph_clusters_i == cluster_id)
                        transform_delta = index_equals_cluster_id[0].tolist()
                        node_ids_for_removal += transform_delta

                if self.gn_debug:
                    print("node_ids_for_removal", node_ids_for_removal)

                if len(node_ids_for_removal) > 0:
                    # Mark invalid nodes
                    valid_nodes_mask_i[node_ids_for_removal] = False

                    # Keep only nodes and edges for valid nodes
                    graph_nodes_i = graph_nodes_i[valid_nodes_mask_i.squeeze()]
                    graph_edges_i = graph_edges_i[valid_nodes_mask_i.squeeze()]
                    graph_edges_weights_i = graph_edges_weights_i[valid_nodes_mask_i.squeeze()]

                    # Update number of nodes
                    optimized_node_count = graph_nodes_i.shape[0]

                    # Get mask of correspondences for which any one of their anchors is an invalid node
                    valid_corresp_mask = torch.ones((match_count), dtype=torch.bool, device=source.device)
                    for node_id_for_removal in node_ids_for_removal:
                        valid_corresp_mask = valid_corresp_mask & torch.all(source_anchors != node_id_for_removal,
                                                                            axis=1)

                    source_points_filtered = source_points_filtered[valid_corresp_mask]
                    source_colors_filtered = source_colors_filtered[valid_corresp_mask]
                    target_matches_filtered = target_matches_filtered[valid_corresp_mask]
                    xy_pixels_warped_filtered = xy_pixels_warped_filtered[valid_corresp_mask]
                    correspondence_weights_filtered = correspondence_weights_filtered[valid_corresp_mask]
                    source_anchors = source_anchors[valid_corresp_mask]
                    source_weights = source_weights[valid_corresp_mask]

                    match_count = source_points_filtered.shape[0]

                    # Update node_ids in edges and anchors by mapping old indices to new indices
                    map_opt_nodes_to_complete_nodes_i = []
                    node_count = 0
                    for node_id, is_node_valid in enumerate(valid_nodes_mask_i):
                        if is_node_valid:
                            graph_edges_i[graph_edges_i == node_id] = node_count
                            source_anchors[source_anchors == node_id] = node_count
                            map_opt_nodes_to_complete_nodes_i.append(node_id)
                            node_count += 1

            if match_count == 0:
                if self.gn_debug:
                    print("\tSolver failed: No valid correspondences")
                convergence_info[i_batch]["errors"].append("Solver failed: No valid correspondences after filtering")
                continue

            if optimized_node_count > self.gn_max_nodes or optimized_node_count < self.gn_min_nodes:
                if self.gn_debug:
                    print("\tSolver failed: Invalid number of nodes: {0}".format(optimized_node_count))
                convergence_info[i_batch]["errors"].append(
                    "Solver failed: Invalid number of nodes: {0}".format(optimized_node_count))
                continue

            # Since source anchor ids need to be long in order to be used as indices,
            # we need to convert them.
            assert torch.all(source_anchors >= 0)
            source_anchors = source_anchors.type(torch.int64)
            # endregion

            ############################################################################################################
            # region Filter invalid edges.
            ############################################################################################################
            node_ids = torch.arange(optimized_node_count, dtype=torch.int32, device=source.device) \
                .view(-1, 1).repeat(1, num_neighbors)  # (opt_num_nodes_i, num_neighbors)
            graph_edge_pairs = torch.cat([node_ids.view(-1, num_neighbors, 1),
                                          graph_edges_i.view(-1, num_neighbors, 1)],
                                         2)  # (opt_num_nodes_i, num_neighbors, 2)

            valid_edges = graph_edges_i >= 0
            valid_edge_indices = torch.where(valid_edges)
            graph_edge_pairs_filtered = graph_edge_pairs[valid_edge_indices[0], valid_edge_indices[1], :].type(
                torch.int64)
            graph_edge_weights_pairs = graph_edges_weights_i[valid_edge_indices[0], valid_edge_indices[1]]

            batch_edge_count = graph_edge_pairs_filtered.shape[0]
            # endregion

            ill_posed_system, residuals, rotations_current, translations_current, gn_point_clouds = \
                self.optimizer.optimize_nodes(
                    match_count, optimized_node_count, batch_edge_count, graph_nodes_i, source_anchors, source_weights,
                    source_points_filtered, source_colors_filtered, correspondence_weights_filtered,
                    xy_pixels_warped_filtered,
                    target_matches_filtered, graph_edge_pairs_filtered, graph_edge_weights_pairs, num_neighbors,
                    fx, fy, cx, cy, convergence_info[i_batch]
                )

            ############################################################################################################
            # region Write the solutions.
            ############################################################################################################
            if not ill_posed_system and torch.isfinite(residuals).all():
                node_rotations[i_batch, map_opt_nodes_to_complete_nodes_i, :, :] = rotations_current.view(
                    optimized_node_count, 3, 3)
                node_translations[i_batch, map_opt_nodes_to_complete_nodes_i, :] = translations_current.view(
                    optimized_node_count, 3)
                deformations_validity[i_batch, map_opt_nodes_to_complete_nodes_i] = 1
                valid_solve[i_batch] = 1
            # endregion
            ############################################################################################################
            # region Warp all valid source points using estimated deformations.
            ############################################################################################################
            if valid_solve[i_batch]:
                # Filter out any invalid pixel anchors, and invalid source points.
                source_points_i = source_points[i_batch].permute(1, 2, 0)
                source_points_i = source_points_i[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(
                    -1, 3, 1)

                # dims: (match_count, 4)
                source_anchors_i = \
                    pixel_anchors[i_batch, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]
                # dims: (match_count, 4)
                source_weights_i = \
                    pixel_weights[i_batch, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]

                num_points = source_points_i.shape[0]

                # Filter out points randomly, if too many are still left.
                if num_points > self.gn_max_warped_points:
                    sampled_indexes = torch.randperm(num_points)[:self.gn_max_warped_points]

                    source_points_i = source_points_i[sampled_indexes]
                    source_anchors_i = source_anchors_i[sampled_indexes]
                    source_weights_i = source_weights_i[sampled_indexes]

                    num_points = self.gn_max_warped_points

                    deformed_points_indices[i_batch] = sampled_indexes
                    deformed_points_subsampled[i_batch] = 1

                source_anchors_i = source_anchors_i.type(torch.int64)

                # Now we deform all source points.
                deformed_points_i = torch.zeros((num_points, 3, 1), dtype=source.dtype, device=source.device)
                graph_nodes_complete_i = graph_nodes[i_batch, :num_nodes_i, :]

                R_final = node_rotations[i_batch, :num_nodes_i, :, :].view(num_nodes_i, 3, 3)
                t_final = node_translations[i_batch, :num_nodes_i, :].view(num_nodes_i, 3, 1)

                for k in range(4):  # Our data uses 4 anchors for every point
                    node_idxs_k = source_anchors_i[:, k]  # (num_points)
                    nodes_k = graph_nodes_complete_i[node_idxs_k].view(num_points, 3, 1)  # (num_points, 3, 1)

                    # Compute deformed point contribution.
                    # (num_points, 3, 1) = (num_points, 3, 3) * (num_points, 3, 1)
                    rotated_points_k = torch.matmul(R_final[node_idxs_k],
                                                    source_points_i - nodes_k)
                    deformed_points_k = rotated_points_k + nodes_k + t_final[node_idxs_k]
                    # (num_points, 3, 1)
                    deformed_points_i += source_weights_i[:, k].view(num_points, 1, 1).repeat(1, 3,
                                                                                              1) * deformed_points_k

                # Store the results.
                deformed_points_prediction[i_batch, :num_points, :] = deformed_points_i.view(1, num_points, 3)
            # endregion
            if valid_solve[i_batch]:
                total_num_matches_per_batch += match_count

            if self.gn_debug:
                if int(valid_solve[i_batch].cpu().numpy()):
                    print("\t\tValid solve   ({:.3f} s)".format(timer() - timer_start))
                else:
                    print("\t\tInvalid solve ({:.3f} s)".format(timer() - timer_start))

            convergence_info[i_batch]["valid"] = int(valid_solve[i_batch].item())
        # endregion
        ###############################################################################################################
        # region We invalidate complete batch if we have too many matches in total (otherwise backprop crashes)
        ###############################################################################################################
        if not evaluate and total_num_matches_per_batch > self.gn_max_matches_train_per_batch:
            error_string = f"Solver failed: too many matches per batch: {total_num_matches_per_batch}"
            print("\t\t" + error_string)
            for i_batch in range(batch_count):
                convergence_info[i_batch]["errors"].append(error_string)
                valid_solve[i_batch] = 0
        # endregion
        return {
            "flow_data": [flow2, flow3, flow4, flow5, flow6],
            "node_rotations": node_rotations,
            "node_translations": node_translations,
            "deformations_validity": deformations_validity,
            "deformed_points_pred": deformed_points_prediction,
            "valid_solve": valid_solve,
            "mask_pred": mask_net_prediction,
            "correspondence_info": [
                xy_coords_warped,
                source_points, valid_source_points,
                target_matches, valid_target_matches,
                correspondence_mask, deformed_points_indices, deformed_points_subsampled
            ],
            "convergence_info": convergence_info,
            "weight_info": weight_info,
            "gn_point_clouds": gn_point_clouds
        }

    def apply_flow_net(self, source_color: torch.Tensor, target_color: torch.Tensor, image_height: int,
                       image_width: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ########################################################################
        # region Compute dense flow from source to target.
        ########################################################################
        flow2, flow3, flow4, flow5, flow6, features2 = self.flow_net.forward(source_color, target_color)

        assert torch.isfinite(flow2).all()
        assert torch.isfinite(features2).all()

        # TODO: explain -- why is there a factor of 20 here?
        flow = 20.0 * torch.nn.functional.interpolate(input=flow2, size=(image_height, image_width), mode='bilinear',
                                                      align_corners=False)
        # endregion
        return flow, flow2, flow3, flow4, flow5, flow6, features2

    def construct_point_to_point_correspondences(self, source_points, target_points, pixel_anchors, xy_coords_warped) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample the target point cloud given the 2D correspondence / flow-field, compile filtering masks
        :param source_points: source 3D point cloud of size (batch_size, 3, height, width)
        :param target_points: target 3D point cloud of size (batch_size, 3, height, width)
        :param pixel_anchors: array of per-pixel anchor node indices, size (batch_size, K, height, width)
        :param xy_coords_warped: estimated 2D correspondence flow-field
        :return: tuple in the form of:
         (valid_source_points, target_matches, valid_target_matches, valid_correspondences, valid_correspondence_counts)
            valid_source_points: boolean array of shape (bs, height, width), where "true" entries correspond to points 
                from the source point cloud whose depth falls within the appropriate depth range, 
                i.e. 0 < depth < max_depth, and all of whose anchors are valid (i.e. nodes have index >= 0)
            target_matches: points from target point cloud sampled at the 2D correspondence flow-field grid
            valid_target_matches: boolean array of shape (bs, height, width), where "true" fields correspond to 
                target_matches points whose depth falls within the appropriate depth range (i.e. 0 < depth < max_depth)
            valid_correspondences: boolean array of shape (bs, height, width) which is a logical combination of
                valid_source_points and valid_target_matches 
        """
        # Construct point-to-point correspondences between source <-> target points.
        # Mask out invalid source points.
        source_anchor_validity = torch.all(pixel_anchors >= 0, dim=3)

        # Sample target points at computed pixel locations.

        target_matches = torch.nn.functional.grid_sample(
            target_points, xy_coords_warped, mode=self._depth_sampling_mode, padding_mode='zeros', align_corners=False
        )

        # We filter out any boundary matches where any of the four pixels are invalid.
        # target_validity: mask for all points that are within the desired depth range
        target_point_validity = ((target_points > 0.0) & (target_points <= self.gn_max_depth)).type(torch.float32)
        target_matches_validity = torch.nn.functional.grid_sample(
            target_point_validity, xy_coords_warped, mode="bilinear", padding_mode='zeros', align_corners=False
        )
        # match validity mask
        target_matches_validity = target_matches_validity[:, 2, :, :] >= 0.999

        # Prepare masks for both valid source points and target matches
        valid_source_points = (source_points[:, 2, :, :] > 0.0) & (
                source_points[:, 2, :, :] <= self.gn_max_depth) & source_anchor_validity
        valid_target_matches = (target_matches[:, 2, :, :] > 0.0) & (
                target_matches[:, 2, :, :] <= self.gn_max_depth) & target_matches_validity
        # Compute mask of valid correspondences
        valid_correspondences = valid_source_points & valid_target_matches
        valid_correspondence_counts = torch.sum(valid_correspondences, dim=(1, 2))
        return valid_source_points, target_matches, valid_target_matches, valid_correspondences, valid_correspondence_counts

    def apply_mask_net(self, evaluate: bool, target_color: torch.Tensor, xy_coords_warped: torch.Tensor,
                       source: torch.Tensor, target_matches: torch.Tensor, features2: torch.Tensor,
                       batch_count: int, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare the input of both the MaskNet and AttentionNet, if we actually use either of them
        target_rgb_warped = torch.nn.functional.grid_sample(target_color, xy_coords_warped, padding_mode='zeros',
                                                            align_corners=False)

        mask_input = torch.cat([source, target_rgb_warped, target_matches], 1)
        mask_net_prediction: torch.Tensor = \
            self.mask_net(features2, mask_input).view(batch_count, image_height, image_width)
        correspondence_weights = mask_net_prediction

        if evaluate:
            # Hard threshold
            if self.threshold_mask_predictions:
                # noinspection PyTypeChecker
                correspondence_weights = torch.where(mask_net_prediction < self.threshold,
                                                     torch.zeros_like(mask_net_prediction), mask_net_prediction)

            # Patch-wise threshold
            elif self.patchwise_threshold_mask_predictions:
                pooled = torch.nn.functional.max_pool2d(input=mask_net_prediction, kernel_size=self.patch_size,
                                                        stride=self.patch_size)
                pooled = torch.nn.functional.interpolate(input=pooled.unsqueeze(1), size=(
                    self.alignment_image_height, self.alignment_image_width),
                                                         mode='nearest').squeeze(1)
                selected = (torch.abs(mask_net_prediction - pooled) <= 1e-8).type(torch.float32)

                correspondence_weights = mask_net_prediction * selected  # * options.patch_size**2
        return mask_net_prediction, correspondence_weights
