import os
from collections import namedtuple

import pytest
import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from alignment import DeformNet
from alignment.default import load_default_nnrt_network

import torch
import open3d.core as o3c

# This test is simply used to ensure (as much as possible) that things are not messed up while we overhaul
# the forward method of DeformNet
from settings import Parameters

AlignmentTestParameters = namedtuple('AlignmentTestParameters', ['config_file_name', 'input_data_folder', 'gt_data_folder'])


@pytest.mark.parametrize("parameters", [AlignmentTestParameters("nnrt_fusion_parameters_flow.yaml", "inputs", "gt_flow"),
                                        AlignmentTestParameters("nnrt_fusion_parameters_test1.yaml", "inputs", "gt_test1")])
def test_alignment_holistic(parameters: AlignmentTestParameters):
    import ext_argparse
    configuration_path = os.path.join(Path(__file__).parent.parent.resolve(), f"configuration_files/{parameters.config_file_name}")
    ext_argparse.process_settings_file(Parameters, configuration_path, generate_default_settings_if_missing=True)

    # make output deterministic
    seed = 1234
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    deform_net: DeformNet = load_default_nnrt_network(o3c.Device.CUDA)

    test_path = Path(__file__).parent.resolve()
    test_data_path = os.path.join(test_path, "test_data")

    # load inputs
    alignment_test_data_path = os.path.join(test_data_path, "alignment_holistic_tests")
    inputs_path = os.path.join(alignment_test_data_path, parameters.input_data_folder)
    gt_path = os.path.join(alignment_test_data_path, parameters.gt_data_folder)

    source_cuda = torch.load(os.path.join(inputs_path, "source_cuda.pt"))
    target_cuda = torch.load(os.path.join(inputs_path, "target_cuda.pt"))
    graph_nodes_cuda = torch.load(os.path.join(inputs_path, "graph_nodes_cuda.pt"))
    graph_edges_cuda = torch.load(os.path.join(inputs_path, "graph_edges_cuda.pt"))
    graph_edges_weights_cuda = torch.load(os.path.join(inputs_path, "graph_edges_weights_cuda.pt"))
    graph_clusters_cuda = torch.load(os.path.join(inputs_path, "graph_clusters_cuda.pt"))
    pixel_anchors_cuda = torch.load(os.path.join(inputs_path, "pixel_anchors_cuda.pt"))
    pixel_weights_cuda = torch.load(os.path.join(inputs_path, "pixel_weights_cuda.pt"))
    node_count_cuda = torch.load(os.path.join(inputs_path, "node_count_cuda.pt"))
    intrinsics_cuda = torch.load(os.path.join(inputs_path, "intrinsics_cuda.pt"))

    with torch.no_grad():
        deform_net_data = deform_net(
            source_cuda, target_cuda,
            graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
            pixel_anchors_cuda, pixel_weights_cuda,
            node_count_cuda, intrinsics_cuda,
            evaluate=True, split="test"
        )

    flow2, flow3, flow4, flow5, flow6 = tuple(deform_net_data["flow_data"])
    node_rotations = deform_net_data["node_rotations"]
    node_translations = deform_net_data["node_translations"]
    deformations_validity = deform_net_data["deformations_validity"]
    deformed_points_pred = deform_net_data["deformed_points_pred"]
    valid_solve = deform_net_data["valid_solve"]
    mask_pred = deform_net_data["mask_pred"]
    # @formatter:off
    xy_coords_warped, source_points, valid_source_points, target_matches, \
        valid_target_matches, valid_correspondences, deformed_points_idxs, deformed_points_subsampled = \
        tuple(deform_net_data["correspondence_info"])
    # @formatter:on

    save_ground_truth = False
    if save_ground_truth:
        torch.save(flow2, os.path.join(gt_path, "flow2.pt"))
        torch.save(flow3, os.path.join(gt_path, "flow3.pt"))
        torch.save(flow4, os.path.join(gt_path, "flow4.pt"))
        torch.save(flow5, os.path.join(gt_path, "flow5.pt"))
        torch.save(flow6, os.path.join(gt_path, "flow6.pt"))
        torch.save(node_rotations, os.path.join(gt_path, "node_rotations.pt"))
        torch.save(node_translations, os.path.join(gt_path, "node_translations.pt"))
        torch.save(deformations_validity, os.path.join(gt_path, "deformations_validity.pt"))
        torch.save(deformed_points_pred, os.path.join(gt_path, "deformed_points_pred.pt"))
        torch.save(valid_solve, os.path.join(gt_path, "valid_solve.pt"))
        torch.save(mask_pred, os.path.join(gt_path, "mask_pred.pt"))
        torch.save(xy_coords_warped, os.path.join(gt_path, "xy_coords_warped.pt"))
        torch.save(source_points, os.path.join(gt_path, "source_points.pt"))
        torch.save(valid_source_points, os.path.join(gt_path, "valid_source_points.pt"))
        torch.save(target_matches, os.path.join(gt_path, "target_matches.pt"))
        torch.save(valid_target_matches, os.path.join(gt_path, "valid_target_matches.pt"))
        torch.save(valid_correspondences, os.path.join(gt_path, "valid_correspondences.pt"))
        torch.save(deformed_points_idxs, os.path.join(gt_path, "deformed_points_idxs.pt"))
        torch.save(deformed_points_subsampled, os.path.join(gt_path, "deformed_points_subsampled.pt"))
    else:
        # load ground truth
        gt_flow2 = torch.load(os.path.join(gt_path, "flow2.pt"))
        gt_flow3 = torch.load(os.path.join(gt_path, "flow3.pt"))
        gt_flow4 = torch.load(os.path.join(gt_path, "flow4.pt"))
        gt_flow5 = torch.load(os.path.join(gt_path, "flow5.pt"))
        gt_flow6 = torch.load(os.path.join(gt_path, "flow6.pt"))
        gt_node_rotations = torch.load(os.path.join(gt_path, "node_rotations.pt"))
        gt_node_translations = torch.load(os.path.join(gt_path, "node_translations.pt"))
        gt_deformations_validity = torch.load(os.path.join(gt_path, "deformations_validity.pt"))
        gt_deformed_points_pred = torch.load(os.path.join(gt_path, "deformed_points_pred.pt"))
        gt_valid_solve = torch.load(os.path.join(gt_path, "valid_solve.pt"))
        gt_mask_pred = torch.load(os.path.join(gt_path, "mask_pred.pt"))
        gt_xy_coords_warped = torch.load(os.path.join(gt_path, "xy_coords_warped.pt"))
        gt_source_points = torch.load(os.path.join(gt_path, "source_points.pt"))
        gt_valid_source_points = torch.load(os.path.join(gt_path, "valid_source_points.pt"))
        gt_target_matches = torch.load(os.path.join(gt_path, "target_matches.pt"))
        gt_valid_target_matches = torch.load(os.path.join(gt_path, "valid_target_matches.pt"))
        gt_valid_correspondences = torch.load(os.path.join(gt_path, "valid_correspondences.pt"))
        gt_deformed_points_idxs = torch.load(os.path.join(gt_path, "deformed_points_idxs.pt"))
        gt_deformed_points_subsampled = torch.load(os.path.join(gt_path, "deformed_points_subsampled.pt"))

        assert torch.equal(flow2, gt_flow2)
        assert torch.equal(flow3, gt_flow3)
        assert torch.equal(flow4, gt_flow4)
        assert torch.equal(flow5, gt_flow5)
        assert torch.equal(flow6, gt_flow6)

        assert torch.equal(node_rotations, gt_node_rotations)
        assert torch.equal(node_translations, gt_node_translations)
        assert torch.equal(deformations_validity, gt_deformations_validity)
        assert torch.equal(deformed_points_pred, gt_deformed_points_pred)
        assert torch.equal(valid_solve, gt_valid_solve)
        assert torch.equal(mask_pred, gt_mask_pred)
        assert torch.equal(xy_coords_warped, gt_xy_coords_warped)
        assert torch.equal(source_points, gt_source_points)
        assert torch.equal(valid_source_points, gt_valid_source_points)
        assert torch.equal(target_matches, gt_target_matches)
        assert torch.equal(valid_target_matches, gt_valid_target_matches)
        assert torch.equal(valid_correspondences, gt_valid_correspondences)
        assert torch.equal(deformed_points_idxs, gt_deformed_points_idxs)
        assert torch.equal(deformed_points_subsampled, gt_deformed_points_subsampled)
