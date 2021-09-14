from alignment import pwcnet
import torch
import torch.nn as nn
import numpy as np
import math
import kornia
from timeit import default_timer as timer

# from settings import settings_general
from alignment.nn_utilities import make_conv_2d, ResBlock2d, Identity


class MaskNet(torch.nn.Module):
    def __init__(self, use_batch_normalization: bool = False):

        super().__init__()

        fn_0 = 16
        self.input_fn = fn_0 + 6 * 2
        fn_1 = 16

        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=565, out_channels=2 * fn_0, kernel_size=4, stride=2, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=2 * fn_0, out_channels=fn_0, kernel_size=4, stride=2, padding=1)

        if use_batch_normalization:
            custom_batch_norm = torch.nn.BatchNorm2d
        else:
            custom_batch_norm = Identity

        self.model = nn.Sequential(
            make_conv_2d(self.input_fn, fn_1, n_blocks=1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            nn.Conv2d(fn_1, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, features, x):
        # Reduce number of channels and upscale to highest resolution
        features = self.upconv1(features)
        features = self.upconv2(features)

        x = torch.cat([features, x], 1)
        assert x.shape[1] == self.input_fn

        return self.model(x)


class LinearSolverLU(torch.autograd.Function):
    """
    LU linear solver.
    """

    @staticmethod
    def forward(ctx, A, b):
        A_LU, pivots = torch.lu(A)
        x = torch.lu_solve(b, A_LU, pivots)

        ctx.save_for_backward(A_LU, pivots, x)

        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_LU, pivots, x = ctx.saved_tensors

        # Math:
        # A * grad_b = grad_x
        # grad_A = -grad_b * x^T

        grad_b = torch.lu_solve(grad_x, A_LU, pivots)
        grad_A = -torch.matmul(grad_b, x.view(1, -1))

        return grad_A, grad_b


class DeformNet(torch.nn.Module):
    def __init__(self,

                 alignment_image_width: int,
                 alignment_image_height: int,

                 skip_solver: bool,

                 freeze_optical_flow_net: bool,
                 freeze_mask_net: bool,

                 gn_depth_sampling_mode: str,
                 gn_max_depth: float,
                 gn_min_nodes: int,
                 gn_max_nodes: int,
                 gn_max_matches_train: int,
                 gn_max_matches_train_per_batch: int,
                 gn_max_matches_eval: int,
                 gn_max_warped_points: int,
                 gn_debug: bool,
                 gn_print_timings: bool,

                 gn_use_edge_weighting: bool,
                 gn_check_condition_num: bool,
                 gn_break_on_condition_num: bool,
                 gn_max_condition_num: float,

                 gn_remove_clusters_with_few_matches: bool,
                 gn_min_num_correspondences_per_cluster: int,

                 gn_num_iter: int,
                 gn_data_flow: float,
                 gn_data_depth: float,
                 gn_arap: float,
                 gn_lm_factor: float,

                 use_mask: bool,

                 threshold_mask_predictions: bool,
                 threshold: float,
                 patchwise_threshold_mask_predictions: bool,
                 patch_size: int,

                 use_batch_norm: bool,

                 telemetry_generator=None):
        super().__init__()

        self.skip_solver = skip_solver

        self.gn_depth_sampling_mode = gn_depth_sampling_mode
        self.gn_max_depth = gn_max_depth
        self.gn_min_nodes = gn_min_nodes
        self.gn_max_nodes = gn_max_nodes
        self.gn_max_matches_train = gn_max_matches_train
        self.gn_max_matches_train_per_batch = gn_max_matches_train_per_batch
        self.gn_max_matches_eval = gn_max_matches_eval
        self.gn_max_warped_points = gn_max_warped_points
        self.gn_debug = gn_debug
        self.gn_print_timings = gn_print_timings

        self.gn_use_edge_weighting = gn_use_edge_weighting
        self.gn_check_condition_num = gn_check_condition_num
        self.gn_break_on_condition_num = gn_break_on_condition_num
        self.gn_max_condition_num = gn_max_condition_num

        self.gn_remove_clusters_with_few_matches = gn_remove_clusters_with_few_matches
        self.gn_min_num_correspondences_per_cluster = gn_min_num_correspondences_per_cluster

        self.gn_num_iter = gn_num_iter
        self.gn_data_flow = gn_data_flow
        self.gn_data_depth = gn_data_depth
        self.gn_arap = gn_arap
        self.gn_lm_factor = gn_lm_factor

        self.alignment_image_width = alignment_image_width
        self.alignment_image_height = alignment_image_height

        self.use_mask = use_mask
        self.threshold_mask_predictions = threshold_mask_predictions
        self.threshold = threshold
        self.patchwise_threshold_mask_predictions = patchwise_threshold_mask_predictions
        self.patch_size = patch_size

        self.telemetry_generator = telemetry_generator

        # Optical flow network
        self.flow_net = pwcnet.PWCNet()
        if freeze_optical_flow_net:
            # Freeze
            for param in self.flow_net.parameters():
                param.requires_grad = False

        # Weight prediction network
        self.mask_net = MaskNet(use_batch_norm)
        if freeze_mask_net:
            # Freeze
            for param in self.mask_net.parameters():
                param.requires_grad = False

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

    def forward(
            self, source, target,
            graph_nodes, graph_edges, graph_edges_weights, graph_clusters,
            pixel_anchors, pixel_weights,
            num_nodes_vec, intrinsics,
            evaluate=False, split="train"
    ):
        batch_size = source.shape[0]

        image_width = source.shape[3]
        image_height = source.shape[2]

        convergence_info = []
        for i in range(batch_size):
            convergence_info.append({
                "total": [],
                "arap": [],
                "data": [],
                "condition_numbers": [],
                "valid": 0,
                "errors": []
            })

        ########################################################################
        # region Compute dense flow from source to target.
        ########################################################################
        source_color = torch.clone(source[:, :3, :, :])  # clones avoid memory bugs here
        target_color = torch.clone(target[:, :3, :, :])
        flow2, flow3, flow4, flow5, flow6, features2 = self.flow_net.forward(source_color, target_color)

        assert torch.isfinite(flow2).all()
        assert torch.isfinite(features2).all()

        flow = 20.0 * torch.nn.functional.interpolate(input=flow2, size=(image_height, image_width), mode='bilinear', align_corners=False)
        # endregion
        ########################################################################
        # region Initialize graph data.
        ########################################################################

        num_nodes_total = graph_nodes.shape[1]
        num_neighbors = graph_edges.shape[2]

        # We assume we always use 4 nearest anchors.
        assert pixel_anchors.shape[3] == 4
        # endregion
        ########################################################################
        # region Apply dense flow to warp the source points to target frame.
        ########################################################################
        x_coords = torch.arange(image_width, dtype=torch.float32, device=source.device).unsqueeze(0).expand(image_height, image_width).unsqueeze(0)
        y_coords = torch.arange(image_height, dtype=torch.float32, device=source.device).unsqueeze(1).expand(image_height, image_width).unsqueeze(0)

        xy_coords = torch.cat([x_coords, y_coords], 0)
        xy_coords = xy_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (bs, 2, 448, 640)

        # Apply the flow to pixel coordinates.
        xy_coords_warped = xy_coords + flow
        xy_pixels_warped = xy_coords_warped.clone()

        # Normalize to be between -1 and 1.
        # Since we use "align_corners=False", the boundaries of corner pixels
        # are -1 and 1, not their centers.
        xy_coords_warped[:, 0, :, :] = (xy_coords_warped[:, 0, :, :]) / (image_width - 1)
        xy_coords_warped[:, 1, :, :] = (xy_coords_warped[:, 1, :, :]) / (image_height - 1)
        xy_coords_warped = xy_coords_warped * 2 - 1

        # Permute the warped coordinates to fit the grid_sample format.
        xy_coords_warped = xy_coords_warped.permute(0, 2, 3, 1)
        # endregion
        ########################################################################
        # region Construct point-to-point correspondences between source <-> target points.
        ########################################################################
        # Mask out invalid source points.
        source_points = source[:, 3:, :, :].clone()
        source_colors = source[:, :3, :, :].clone()
        source_anchor_validity = torch.all(pixel_anchors >= 0.0, dim=3)

        # Sample target points at computed pixel locations.
        target_points = target[:, 3:, :, :].clone()
        target_matches = torch.nn.functional.grid_sample(
            target_points, xy_coords_warped, mode=self.gn_depth_sampling_mode, padding_mode='zeros', align_corners=False
        )

        # We filter out any boundary matches where any of the four pixels are invalid.
        # target_validity: mask for all points that are within the desired depth range
        target_validity = ((target_points > 0.0) & (target_points <= self.gn_max_depth)).type(torch.float32)
        target_matches_validity = torch.nn.functional.grid_sample(
            target_validity, xy_coords_warped, mode="bilinear", padding_mode='zeros', align_corners=False
        )
        # match validity mask
        target_matches_validity = target_matches_validity[:, 2, :, :] >= 0.999

        # Prepare masks for both valid source points and target matches
        valid_source_points = (source_points[:, 2, :, :] > 0.0) & (
                source_points[:, 2, :, :] <= self.gn_max_depth) & source_anchor_validity
        valid_target_matches = (target_matches[:, 2, :, :] > 0.0) & (
                target_matches[:, 2, :, :] <= self.gn_max_depth) & target_matches_validity

        # Prepare the input of both the MaskNet and AttentionNet, if we actually use either of them
        if self.use_mask:
            target_rgb = target[:, :3, :, :].clone()
            target_rgb_warped = torch.nn.functional.grid_sample(target_rgb, xy_coords_warped, padding_mode='zeros', align_corners=False)

            mask_input = torch.cat([source, target_rgb_warped, target_matches], 1)
        # endregion
        ########################################################################
        # region MaskNet
        ########################################################################
        # We predict correspondence weights [0, 1], if we use mask network.
        mask_pred = None
        if self.use_mask:
            mask_pred = self.mask_net(features2, mask_input).view(batch_size, image_height, image_width)

        # Compute mask of valid correspondences
        valid_correspondences = valid_source_points & valid_target_matches

        # Initialize correspondence weights with 1's. We might overwrite them with MaskNet-predicted weights next
        correspondence_weights = torch.ones(valid_target_matches.shape, dtype=torch.float32, device=valid_target_matches.device)

        # We invalidate target matches later, when we assign a weight to each match.
        if self.use_mask:
            correspondence_weights = mask_pred

            if evaluate:
                # Hard threshold
                if self.threshold_mask_predictions:
                    correspondence_weights = torch.where(mask_pred < self.threshold, torch.zeros_like(mask_pred), mask_pred)

                # Patch-wise threshold
                elif self.patchwise_threshold_mask_predictions:
                    pooled = torch.nn.functional.max_pool2d(input=mask_pred, kernel_size=self.patch_size,
                                                            stride=self.patch_size)
                    pooled = torch.nn.functional.interpolate(input=pooled.unsqueeze(1), size=(
                        self.alignment_image_height, self.alignment_image_width),
                                                             mode='nearest').squeeze(1)
                    selected = (torch.abs(mask_pred - pooled) <= 1e-8).type(torch.float32)

                    correspondence_weights = mask_pred * selected  # * options.patch_size**2

        num_valid_correspondences = torch.sum(valid_correspondences, dim=(1, 2))
        # endregion
        ########################################################################
        # region Initialize graph data.
        ########################################################################
        num_nodes_total = graph_nodes.shape[1]
        num_neighbors = graph_edges.shape[2]

        # We assume we always use 4 nearest anchors.
        assert pixel_anchors.shape[3] == 4

        node_rotations = torch.eye(3, dtype=source.dtype, device=source.device).view(1, 1, 3, 3).repeat(batch_size, num_nodes_total, 1, 1)
        node_translations = torch.zeros((batch_size, num_nodes_total, 3), dtype=source.dtype, device=source.device)
        deformations_validity = torch.zeros((batch_size, num_nodes_total), dtype=source.dtype, device=source.device)
        valid_solve = torch.zeros((batch_size), dtype=torch.uint8, device=source.device)
        deformed_points_pred = torch.zeros((batch_size, self.gn_max_warped_points, 3), dtype=source.dtype, device=source.device)
        deformed_points_idxs = torch.zeros((batch_size, self.gn_max_warped_points), dtype=torch.int64, device=source.device)
        deformed_points_subsampled = torch.zeros((batch_size), dtype=torch.uint8, device=source.device)

        # Optionally, skip the solver
        if not evaluate and self.skip_solver:
            return {
                "flow_data": [flow2, flow3, flow4, flow5, flow6],
                "node_rotations": node_rotations,
                "node_translations": node_translations,
                "deformations_validity": deformations_validity,
                "deformed_points_pred": deformed_points_pred,
                "valid_solve": valid_solve,
                "mask_pred": mask_pred,
                "correspondence_info": [
                    xy_coords_warped,
                    source_points, valid_source_points,
                    target_matches, valid_target_matches,
                    None, deformed_points_idxs, deformed_points_subsampled
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
        correspondences_exist = num_valid_correspondences > 0

        total_num_matches_per_batch = 0

        weight_info = {
            "total_corres_num": 0,
            "total_corres_weight": 0.0
        }

        self.vec_to_skew_mat.to(source.device)

        for i in range(batch_size):
            if self.gn_debug:
                print()
                print("--Sample", i, "in batch--")

            num_nodes_i = num_nodes_vec[i]
            if num_nodes_i > self.gn_max_nodes or num_nodes_i < self.gn_min_nodes:
                print("\tSolver failed: Invalid number of canonical_node_positions: {0}".format(num_nodes_i))
                convergence_info[i]["errors"].append("Solver failed: Invalid number of canonical_node_positions: {0}".format(num_nodes_i))
                continue

            if not correspondences_exist[i]:
                print("\tSolver failed: No valid correspondences before filtering")
                convergence_info[i]["errors"].append("Solver failed: No valid correspondences before filtering")
                continue

            timer_start = timer()

            graph_nodes_i = graph_nodes[i, :num_nodes_i, :]  # (num_nodes_i, 3)
            graph_edges_i = graph_edges[i, :num_nodes_i, :]  # (num_nodes_i, 8)
            graph_edges_weights_i = graph_edges_weights[i, :num_nodes_i, :]  # (num_nodes_i, 8)
            graph_clusters_i = graph_clusters[i, :num_nodes_i, :]  # (num_nodes_i, 1)

            fx = intrinsics[i, 0]
            fy = intrinsics[i, 1]
            cx = intrinsics[i, 2]
            cy = intrinsics[i, 3]

            ###############################################################################################################
            # region Filter invalid matches.
            ###############################################################################################################
            valid_correspondences_idxs = torch.where(valid_correspondences[i])

            source_points_filtered = source_points[i].permute(1, 2, 0)
            source_points_filtered = source_points_filtered[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 3, 1)
            source_colors_filtered = source_colors[i].permute(1, 2, 0)
            source_colors_filtered = source_colors_filtered[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 3, 1)

            target_matches_filtered = target_matches[i].permute(1, 2, 0)
            target_matches_filtered = target_matches_filtered[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 3, 1)

            xy_pixels_warped_filtered = xy_pixels_warped[i].permute(1, 2, 0)  # (height, width, 2)
            xy_pixels_warped_filtered = xy_pixels_warped_filtered[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 2, 1)

            correspondence_weights_filtered = correspondence_weights[i, valid_correspondences_idxs[0], valid_correspondences_idxs[1]].view(
                -1)  # (num_matches)

            source_anchors = pixel_anchors[i, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]  # (num_matches, 4)
            source_weights = pixel_weights[i, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]  # (num_matches, 4)

            num_matches = source_points_filtered.shape[0]
            # endregion
            ###############################################################################################################
            # region Generate weight info to estimate average weight.
            ###############################################################################################################
            weight_info = {
                "total_corres_num": correspondence_weights_filtered.shape[0],
                "total_corres_weight": float(correspondence_weights_filtered.sum())
            }
            # endregion
            ###############################################################################################################
            # region Randomly subsample matches, if necessary.
            ###############################################################################################################
            if split == "val" or split == "test":
                max_num_matches = self.gn_max_matches_eval
            elif split == "train":
                max_num_matches = self.gn_max_matches_train
            else:
                raise Exception("Split {} is not defined".format(split))

            if num_matches > max_num_matches:
                sampled_idxs = torch.randperm(num_matches)[:max_num_matches]
                source_points_filtered = source_points_filtered[sampled_idxs]
                source_colors_filtered = source_colors_filtered[sampled_idxs]
                target_matches_filtered = target_matches_filtered[sampled_idxs]
                xy_pixels_warped_filtered = xy_pixels_warped_filtered[sampled_idxs]
                correspondence_weights_filtered = correspondence_weights_filtered[sampled_idxs]
                source_anchors = source_anchors[sampled_idxs]
                source_weights = source_weights[sampled_idxs]

                num_matches = max_num_matches
            # endregion
            ###############################################################################################################
            # region Remove canonical_node_positions if their corresponding clusters don't have enough correspondences
            # (Correspondences that have "bad" canonical_node_positions as anchors will also be removed)
            ###############################################################################################################
            map_opt_nodes_to_complete_nodes_i = list(range(0, num_nodes_i))
            opt_num_nodes_i = num_nodes_i

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

                # we'll build a mask that stores which canonical_node_positions will survive
                valid_nodes_mask_i = torch.ones((num_nodes_i), dtype=torch.bool, device=source.device)

                # if not enough matches in a cluster, mark all cluster's canonical_node_positions for removal
                node_ids_for_removal = []
                for cluster_id, cluster_match_weights in match_weights_per_cluster.items():
                    if self.gn_debug:
                        print('cluster_id', cluster_id, cluster_match_weights)

                    if cluster_match_weights < self.gn_min_num_correspondences_per_cluster:
                        index_equals_cluster_id = torch.where(graph_clusters_i == cluster_id)
                        x = index_equals_cluster_id[0].tolist()
                        node_ids_for_removal += x

                if self.gn_debug:
                    print("node_ids_for_removal", node_ids_for_removal)

                if len(node_ids_for_removal) > 0:
                    # Mark invalid canonical_node_positions
                    valid_nodes_mask_i[node_ids_for_removal] = False

                    # Kepp only canonical_node_positions and edges for valid canonical_node_positions
                    graph_nodes_i = graph_nodes_i[valid_nodes_mask_i.squeeze()]
                    graph_edges_i = graph_edges_i[valid_nodes_mask_i.squeeze()]
                    graph_edges_weights_i = graph_edges_weights_i[valid_nodes_mask_i.squeeze()]

                    # Update number of canonical_node_positions
                    opt_num_nodes_i = graph_nodes_i.shape[0]

                    # Get mask of correspondences for which any one of their anchors is an invalid node
                    valid_corresp_mask = torch.ones((num_matches), dtype=torch.bool, device=source.device)
                    for node_id_for_removal in node_ids_for_removal:
                        valid_corresp_mask = valid_corresp_mask & torch.all(source_anchors != node_id_for_removal, axis=1)

                    source_points_filtered = source_points_filtered[valid_corresp_mask]
                    source_colors_filtered = source_colors_filtered[valid_corresp_mask]
                    target_matches_filtered = target_matches_filtered[valid_corresp_mask]
                    xy_pixels_warped_filtered = xy_pixels_warped_filtered[valid_corresp_mask]
                    correspondence_weights_filtered = correspondence_weights_filtered[valid_corresp_mask]
                    source_anchors = source_anchors[valid_corresp_mask]
                    source_weights = source_weights[valid_corresp_mask]

                    num_matches = source_points_filtered.shape[0]

                    # Update node_ids in edges and anchors by mapping old indices to new indices
                    map_opt_nodes_to_complete_nodes_i = []
                    node_count = 0
                    for node_id, is_node_valid in enumerate(valid_nodes_mask_i):
                        if is_node_valid:
                            graph_edges_i[graph_edges_i == node_id] = node_count
                            source_anchors[source_anchors == node_id] = node_count
                            map_opt_nodes_to_complete_nodes_i.append(node_id)
                            node_count += 1

            if num_matches == 0:
                if self.gn_debug:
                    print("\tSolver failed: No valid correspondences")
                convergence_info[i]["errors"].append("Solver failed: No valid correspondences after filtering")
                continue

            if opt_num_nodes_i > self.gn_max_nodes or opt_num_nodes_i < self.gn_min_nodes:
                if self.gn_debug:
                    print("\tSolver failed: Invalid number of canonical_node_positions: {0}".format(opt_num_nodes_i))
                convergence_info[i]["errors"].append("Solver failed: Invalid number of canonical_node_positions: {0}".format(opt_num_nodes_i))
                continue

            # Since source anchor ids need to be long in order to be used as indices,
            # we need to convert them.
            assert torch.all(source_anchors >= 0)
            source_anchors = source_anchors.type(torch.int64)
            # endregion
            ###############################################################################################################
            # region Filter invalid edges.
            ###############################################################################################################
            node_ids = torch.arange(opt_num_nodes_i, dtype=torch.int32, device=source.device).view(-1, 1).repeat(1,
                                                                                                                 num_neighbors)  # (opt_num_nodes_i, num_neighbors)
            graph_edge_pairs = torch.cat([node_ids.view(-1, num_neighbors, 1), graph_edges_i.view(-1, num_neighbors, 1)],
                                         2)  # (opt_num_nodes_i, num_neighbors, 2)

            valid_edges = graph_edges_i >= 0
            valid_edge_idxs = torch.where(valid_edges)
            graph_edge_pairs_filtered = graph_edge_pairs[valid_edge_idxs[0], valid_edge_idxs[1], :].type(torch.int64)
            graph_edge_weights_pairs = graph_edges_weights_i[valid_edge_idxs[0], valid_edge_idxs[1]]

            num_edges_i = graph_edge_pairs_filtered.shape[0]
            # endregion
            ###############################################################################################################
            # region Execute Gauss-Newton solver.
            ###############################################################################################################
            num_gn_iter = self.gn_num_iter
            lambda_data_flow = math.sqrt(self.gn_data_flow)
            lambda_data_depth = math.sqrt(self.gn_data_depth)
            lambda_arap = math.sqrt(self.gn_arap)
            lm_factor = self.gn_lm_factor

            # The parameters in GN solver are 3 parameters for rotation and 3 parameters for
            # translation for every node. All node rotation parameters are listed first, and
            # then all node translation parameters are listed.
            #                        x = [R_current_all, t_current_all]
            R_current = torch.eye(3, dtype=source.dtype, device=source.device).view(1, 3, 3).repeat(opt_num_nodes_i, 1, 1)
            t_current = torch.zeros((opt_num_nodes_i, 3, 1), dtype=source.dtype, device=source.device)

            if self.gn_debug:
                print(
                    "\tNum. matches: {0} || Num. canonical_node_positions: {1} || Num. edges: {2}".format(num_matches, opt_num_nodes_i, num_edges_i))

            # Helper structures.
            data_increment_vec_0_3 = torch.arange(0, num_matches * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_matches)
            data_increment_vec_1_3 = torch.arange(1, num_matches * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_matches)
            data_increment_vec_2_3 = torch.arange(2, num_matches * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_matches)

            if num_edges_i > 0:
                arap_increment_vec_0_3 = torch.arange(0, num_edges_i * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_edges_i)
                arap_increment_vec_1_3 = torch.arange(1, num_edges_i * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_edges_i)
                arap_increment_vec_2_3 = torch.arange(2, num_edges_i * 3, 3, out=torch.cuda.LongTensor(), device=source.device)  # (num_edges_i)
                arap_one_vec = torch.ones((num_edges_i), dtype=source.dtype, device=source.device)

            ill_posed_system = False

            for gn_i in range(num_gn_iter):
                timer_data_start = timer()

                ##########################################
                # region Compute data residual and jacobian.
                ##########################################
                jacobian_data = torch.zeros((num_matches * 3, opt_num_nodes_i * 6), dtype=source.dtype,
                                            device=source.device)  # (num_matches*3, opt_num_nodes_i*6)
                deformed_points = torch.zeros((num_matches, 3, 1), dtype=source.dtype, device=source.device)

                for k in range(4):  # Our data uses 4 anchors for every point
                    node_idxs_k = source_anchors[:, k]  # (num_matches)
                    nodes_k = graph_nodes_i[node_idxs_k].view(num_matches, 3, 1)  # (num_matches, 3, 1)

                    # Compute deformed point contribution.
                    # (num_matches, 3, 1) = (num_matches, 3, 3) * (num_matches, 3, 1)
                    rotated_points_k = torch.matmul(R_current[node_idxs_k],
                                                    source_points_filtered - nodes_k)
                    deformed_points_k = rotated_points_k + nodes_k + t_current[node_idxs_k]
                    deformed_points += \
                        source_weights[:, k].view(num_matches, 1, 1).repeat(1, 3, 1) * \
                        deformed_points_k  # (num_matches, 3, 1)

                if self.telemetry_generator is not None:
                    self.telemetry_generator.process_gn_point_cloud(deformed_points, source_colors_filtered, gn_i)

                # Get necessary components of deformed points.
                eps = 1e-7  # Just as good practice, although matches should all have valid depth at this stage

                deformed_x = deformed_points[:, 0, :].view(num_matches)  # (num_matches)
                deformed_y = deformed_points[:, 1, :].view(num_matches)  # (num_matches)
                deformed_z_inverse = torch.div(1.0, deformed_points[:, 2, :].view(num_matches) + eps)  # (num_matches)
                fx_mul_x = fx * deformed_x  # (num_matches)
                fy_mul_y = fy * deformed_y  # (num_matches)
                fx_div_z = fx * deformed_z_inverse  # (num_matches)
                fy_div_z = fy * deformed_z_inverse  # (num_matches)
                fx_mul_x_div_z = fx_mul_x * deformed_z_inverse  # (num_matches)
                fy_mul_y_div_z = fy_mul_y * deformed_z_inverse  # (num_matches)
                minus_fx_mul_x_div_z_2 = -fx_mul_x_div_z * deformed_z_inverse  # (num_matches)
                minus_fy_mul_y_div_z_2 = -fy_mul_y_div_z * deformed_z_inverse  # (num_matches)

                for k in range(4):  # Our data uses 4 anchors for every point
                    node_idxs_k = source_anchors[:, k]  # (num_matches)
                    nodes_k = graph_nodes_i[node_idxs_k].view(num_matches, 3, 1)  # (num_matches, 3, 1)

                    weights_k = source_weights[:, k] * correspondence_weights_filtered  # (num_matches) #TODO: check arm pixel correspondence_weights

                    # Compute skew-symmetric part.
                    rotated_points_k = torch.matmul(R_current[node_idxs_k],
                                                    source_points_filtered - nodes_k)  # (num_matches, 3, 1) = (num_matches, 3, 3) * (num_matches, 3, 1)
                    weighted_rotated_points_k = weights_k.view(num_matches, 1, 1).repeat(1, 3, 1) * rotated_points_k  # (num_matches, 3, 1)
                    skew_symetric_mat_data = -torch.matmul(self.vec_to_skew_mat, weighted_rotated_points_k).view(num_matches, 3,
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
                    jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 0] += lambda_data_depth * skew_symetric_mat_data[:, 2, 0]
                    jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 1] += lambda_data_depth * skew_symetric_mat_data[:, 2, 1]
                    jacobian_data[data_increment_vec_2_3, 3 * node_idxs_k + 2] += lambda_data_depth * skew_symetric_mat_data[:, 2, 2]

                    assert torch.isfinite(jacobian_data).all(), jacobian_data

                res_data = torch.zeros((num_matches * 3, 1), dtype=source.dtype, device=source.device)

                # FLOW PART
                res_data[data_increment_vec_0_3, 0] = lambda_data_flow * correspondence_weights_filtered * (
                        fx_mul_x_div_z + cx - xy_pixels_warped_filtered[:, 0, :].view(num_matches))
                res_data[data_increment_vec_1_3, 0] = lambda_data_flow * correspondence_weights_filtered * (
                        fy_mul_y_div_z + cy - xy_pixels_warped_filtered[:, 1, :].view(num_matches))

                # DEPTH PART
                res_data[data_increment_vec_2_3, 0] = lambda_data_depth * correspondence_weights_filtered * (
                        deformed_points[:, 2, :] - target_matches_filtered[:, 2, :]).view(num_matches)

                if self.gn_print_timings: print("\t\tData term: {:.3f} s".format(timer() - timer_data_start))
                timer_arap_start = timer()
                # endregion
                ##########################################
                # region Compute arap residual and jacobian.
                ##########################################
                if num_edges_i > 0:
                    jacobian_arap = torch.zeros((num_edges_i * 3, opt_num_nodes_i * 6), dtype=source.dtype,
                                                device=source.device)  # (num_edges_i*3, opt_num_nodes_i*6)

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

                    nodes_0 = graph_nodes_i[node_idxs_0].view(num_edges_i, 3, 1)
                    nodes_1 = graph_nodes_i[node_idxs_1].view(num_edges_i, 3, 1)

                    # Compute residual.
                    rotated_node_delta = torch.matmul(R_current[node_idxs_0], nodes_1 - nodes_0)  # (num_edges_i, 3)
                    res_arap = lambda_arap * w_repeat * (rotated_node_delta + nodes_0 + t_current[node_idxs_0] - (nodes_1 + t_current[node_idxs_1]))
                    res_arap = res_arap.view(num_edges_i * 3, 1)

                    # Compute jacobian wrt. translations.
                    jacobian_arap[
                        arap_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 0] += lambda_arap * w * arap_one_vec  # (num_edges_i)
                    jacobian_arap[
                        arap_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 1] += lambda_arap * w * arap_one_vec  # (num_edges_i)
                    jacobian_arap[
                        arap_increment_vec_2_3, 3 * opt_num_nodes_i + 3 * node_idxs_0 + 2] += lambda_arap * w * arap_one_vec  # (num_edges_i)

                    jacobian_arap[
                        arap_increment_vec_0_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 0] += -lambda_arap * w * arap_one_vec  # (num_edges_i)
                    jacobian_arap[
                        arap_increment_vec_1_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 1] += -lambda_arap * w * arap_one_vec  # (num_edges_i)
                    jacobian_arap[
                        arap_increment_vec_2_3, 3 * opt_num_nodes_i + 3 * node_idxs_1 + 2] += -lambda_arap * w * arap_one_vec  # (num_edges_i)

                    # Compute jacobian wrt. rotations.
                    # Derivative wrt. R_1 is equal to 0.
                    skew_symetric_mat_arap = -lambda_arap * w_repeat_repeat * torch.matmul(self.vec_to_skew_mat, rotated_node_delta).view(num_edges_i,
                                                                                                                                          3,
                                                                                                                                          3)  # (num_edges_i, 3, 3)

                    jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 0] += skew_symetric_mat_arap[:, 0, 0]
                    jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 1] += skew_symetric_mat_arap[:, 0, 1]
                    jacobian_arap[arap_increment_vec_0_3, 3 * node_idxs_0 + 2] += skew_symetric_mat_arap[:, 0, 2]
                    jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 0] += skew_symetric_mat_arap[:, 1, 0]
                    jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 1] += skew_symetric_mat_arap[:, 1, 1]
                    jacobian_arap[arap_increment_vec_1_3, 3 * node_idxs_0 + 2] += skew_symetric_mat_arap[:, 1, 2]
                    jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 0] += skew_symetric_mat_arap[:, 2, 0]
                    jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 1] += skew_symetric_mat_arap[:, 2, 1]
                    jacobian_arap[arap_increment_vec_2_3, 3 * node_idxs_0 + 2] += skew_symetric_mat_arap[:, 2, 2]

                    assert torch.isfinite(jacobian_arap).all(), jacobian_arap

                if self.gn_print_timings: print("\t\tARAP term: {:.3f} s".format(timer() - timer_arap_start))
                # endregion
                ##########################################
                # region Solve linear system.
                ##########################################
                if num_edges_i > 0:
                    res = torch.cat((res_data, res_arap), 0)
                    jac = torch.cat((jacobian_data, jacobian_arap), 0)
                else:
                    res = res_data
                    jac = jacobian_data

                timer_system_start = timer()

                # Compute A = J^TJ and b = -J^Tr.
                jac_t = torch.transpose(jac, 0, 1)
                A = torch.matmul(jac_t, jac)
                b = torch.matmul(-jac_t, res)

                # Solve linear system Ax = b.
                A = A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * lm_factor

                assert torch.isfinite(A).all(), A

                if self.gn_print_timings: print("\t\tSystem computation: {:.3f} s".format(timer() - timer_system_start))
                timer_cond_start = timer()

                # Check the determinant/condition number.
                # If unstable, we break optimization.
                if self.gn_check_condition_num:
                    with torch.no_grad():
                        # Condition number.
                        values, _ = torch.eig(A)
                        real_values = values[:, 0]
                        assert torch.isfinite(real_values).all(), real_values
                        max_eig_value = torch.max(torch.abs(real_values))
                        min_eig_value = torch.min(torch.abs(real_values))
                        condition_number = max_eig_value / min_eig_value
                        condition_number = condition_number.item()
                        convergence_info[i]["condition_numbers"].append(condition_number)

                        if self.gn_break_on_condition_num and (
                                not math.isfinite(condition_number) or condition_number > self.gn_max_condition_num):
                            print("\t\tToo high condition number: {0:e} (max: {1:.3f}, min: {2:.3f}). Discarding sample".format(condition_number,
                                                                                                                                max_eig_value.item(),
                                                                                                                                min_eig_value.item()))
                            convergence_info[i]["errors"].append(
                                "Too high condition number: {0:e} (max: {1:.3f}, min: {2:.3f}). Discarding sample".format(condition_number,
                                                                                                                          max_eig_value.item(),
                                                                                                                          min_eig_value.item()))
                            ill_posed_system = True
                            break
                        elif self.gn_debug:
                            print("\t\tCondition number: {0:e} (max: {1:.3f}, min: {2:.3f})".format(condition_number, max_eig_value.item(),
                                                                                                    min_eig_value.item()))

                if self.gn_print_timings: print("\t\tComputation of cond. num.: {:.3f} s".format(timer() - timer_cond_start))
                timer_solve_start = timer()

                linear_solver = LinearSolverLU.apply

                try:
                    x = linear_solver(A, b)

                except RuntimeError as e:
                    ill_posed_system = True
                    print("\t\tSolver failed: Ill-posed system!", e)
                    convergence_info[i]["errors"].append("Solver failed: Ill-posed system!")
                    break

                if not torch.isfinite(x).all():
                    ill_posed_system = True
                    print("\t\tSolver failed: Non-finite solution x!")
                    convergence_info[i]["errors"].append("Solver failed: Non-finite solution x!")
                    break

                if self.gn_print_timings: print("\t\tLinear solve: {:.3f} s".format(timer() - timer_solve_start))

                # Increment the current rotation and translation.
                R_inc = kornia.angle_axis_to_rotation_matrix(x[:opt_num_nodes_i * 3].view(opt_num_nodes_i, 3))
                t_inc = x[opt_num_nodes_i * 3:].view(opt_num_nodes_i, 3, 1)

                R_current = torch.matmul(R_inc, R_current)
                t_current = t_current + t_inc

                loss_data = torch.norm(res_data).item()
                loss_total = torch.norm(res).item()

                convergence_info[i]["data"].append(loss_data)
                convergence_info[i]["total"].append(loss_total)

                if num_edges_i > 0:
                    loss_arap = torch.norm(res_arap).item()
                    convergence_info[i]["arap"].append(loss_arap)

                if self.gn_debug:
                    if num_edges_i > 0:
                        print("\t\t-->Iteration: {0}. Loss: \tdata = {1:.3f}, \tarap = {2:.3f}, \ttotal = {3:.3f}".format(gn_i, loss_data, loss_arap,
                                                                                                                          loss_total))
                    else:
                        print("\t\t-->Iteration: {0}. Loss: \tdata = {1:.3f}, \ttotal = {2:.3f}".format(gn_i, loss_data, loss_total))
                # endregion
            # endregion
            ###############################################################################################################
            # region Write the solutions.
            ###############################################################################################################
            if not ill_posed_system and torch.isfinite(res).all():
                node_rotations[i, map_opt_nodes_to_complete_nodes_i, :, :] = R_current.view(opt_num_nodes_i, 3, 3)
                node_translations[i, map_opt_nodes_to_complete_nodes_i, :] = t_current.view(opt_num_nodes_i, 3)
                deformations_validity[i, map_opt_nodes_to_complete_nodes_i] = 1
                valid_solve[i] = 1
            # endregion
            ###############################################################################################################
            # region Warp all valid source points using estimated deformations.
            ###############################################################################################################
            if valid_solve[i]:
                # Filter out any invalid pixel anchors, and invalid source points.
                source_points_i = source_points[i].permute(1, 2, 0)
                source_points_i = source_points_i[valid_correspondences_idxs[0], valid_correspondences_idxs[1], :].view(-1, 3, 1)

                source_anchors_i = pixel_anchors[i, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]  # (num_matches, 4)
                source_weights_i = pixel_weights[i, valid_correspondences_idxs[0], valid_correspondences_idxs[1], :]  # (num_matches, 4)

                num_points = source_points_i.shape[0]

                # Filter out points randomly, if too many are still left.
                if num_points > self.gn_max_warped_points:
                    sampled_idxs = torch.randperm(num_points)[:self.gn_max_warped_points]

                    source_points_i = source_points_i[sampled_idxs]
                    source_anchors_i = source_anchors_i[sampled_idxs]
                    source_weights_i = source_weights_i[sampled_idxs]

                    num_points = self.gn_max_warped_points

                    deformed_points_idxs[i] = sampled_idxs
                    deformed_points_subsampled[i] = 1

                source_anchors_i = source_anchors_i.type(torch.int64)

                # Now we deform all source points.
                deformed_points_i = torch.zeros((num_points, 3, 1), dtype=source.dtype, device=source.device)
                graph_nodes_complete_i = graph_nodes[i, :num_nodes_i, :]

                R_final = node_rotations[i, :num_nodes_i, :, :].view(num_nodes_i, 3, 3)
                t_final = node_translations[i, :num_nodes_i, :].view(num_nodes_i, 3, 1)

                for k in range(4):  # Our data uses 4 anchors for every point
                    node_idxs_k = source_anchors_i[:, k]  # (num_points)
                    nodes_k = graph_nodes_complete_i[node_idxs_k].view(num_points, 3, 1)  # (num_points, 3, 1)

                    # Compute deformed point contribution.
                    # (num_points, 3, 1) = (num_points, 3, 3) * (num_points, 3, 1)
                    rotated_points_k = torch.matmul(R_final[node_idxs_k],
                                                    source_points_i - nodes_k)
                    deformed_points_k = rotated_points_k + nodes_k + t_final[node_idxs_k]
                    # (num_points, 3, 1)
                    deformed_points_i += source_weights_i[:, k].view(num_points, 1, 1).repeat(1, 3, 1) * deformed_points_k

                # Store the results.
                deformed_points_pred[i, :num_points, :] = deformed_points_i.view(1, num_points, 3)
            # endregion
            if valid_solve[i]:
                total_num_matches_per_batch += num_matches

            if self.gn_debug:
                if int(valid_solve[i].cpu().numpy()):
                    print("\t\tValid solve   ({:.3f} s)".format(timer() - timer_start))
                else:
                    print("\t\tInvalid solve ({:.3f} s)".format(timer() - timer_start))

            convergence_info[i]["valid"] = int(valid_solve[i].item())
        # endregion
        ###############################################################################################################
        # region We invalidate complete batch if we have too many matches in total (otherwise backprop crashes)
        ###############################################################################################################
        if not evaluate and total_num_matches_per_batch > self.gn_max_matches_train_per_batch:
            print("\t\tSolver failed: Too many matches per batch: {}".format(total_num_matches_per_batch))
            for i in range(batch_size):
                convergence_info[i]["errors"].append("Solver failed: Too many matches per batch: {}".format(total_num_matches_per_batch))
                valid_solve[i] = 0
        # endregion
        return {
            "flow_data": [flow2, flow3, flow4, flow5, flow6],
            "node_rotations": node_rotations,
            "node_translations": node_translations,
            "deformations_validity": deformations_validity,
            "deformed_points_pred": deformed_points_pred,
            "valid_solve": valid_solve,
            "mask_pred": mask_pred,
            "correspondence_info": [
                xy_coords_warped,
                source_points, valid_source_points,
                target_matches, valid_target_matches,
                valid_correspondences, deformed_points_idxs, deformed_points_subsampled
            ],
            "convergence_info": convergence_info,
            "weight_info": weight_info
        }
