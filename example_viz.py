#!/usr/bin/python3
import os

import torch
import numpy as np

from data.camera import load_intrinsic_matrix_entries_as_dict_from_text_4x4_matrix
import utils.image
import utils.viz.tracking as tracking_viz

from alignment import DeformNet

import options

from data import FramePairDataset, FramePairPreset, DeformDataset


# TODO: all of the original NNRT code is suffering from major cases of the long-parameter-list code smell
#  (See https://refactoring.guru/smells/long-parameter-list for reasons, downsides, and refactoring solutions)
#  Through better OO design and refactoring, these should be grouped into objects, replaced with internal
#  method calls, etc.

# TODO: all of the original NNRT code is suffering from major cases of the God-function antipattern, i.e.
#  the ~700-line-long forward function in DeformNet and the like. These god-functions should be split up
#  into sub-functions that "do only one thing"
def main():
    #####################################################################################################
    # Options
    #####################################################################################################

    # Source-target example
    frame_pair_preset: FramePairPreset = FramePairPreset.RED_SHORTS_200_400
    frame_pair_name = frame_pair_preset.name.lower()
    frame_pair_dataset: FramePairDataset = frame_pair_preset.value

    save_node_transformations = False

    source_frame_index = frame_pair_dataset.source_frame_index
    target_frame_index = frame_pair_dataset.target_frame_index
    segment_name = frame_pair_dataset.segment_name

    # We will overwrite the default value in options.py / settings.py
    options.use_mask = True

    #####################################################################################################
    # Load alignment
    #####################################################################################################

    saved_model = options.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct alignment
    model = DeformNet().cuda()

    if "chairs_things" in saved_model:
        model.flow_net.load_state_dict(pretrained_dict)
    else:
        if options.model_module_to_load == "full_model":
            # Load completely alignment
            model.load_state_dict(pretrained_dict)
        elif options.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            print(options.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    model.eval()

    #####################################################################################################
    # Load example dataset
    #####################################################################################################
    intrinsics = load_intrinsic_matrix_entries_as_dict_from_text_4x4_matrix(frame_pair_dataset.get_intrinsics_path())

    image_height = options.alignment_image_height
    image_width = options.alignment_image_width
    max_boundary_distance = options.max_boundary_dist

    src_color_image_path = frame_pair_dataset.get_source_color_image_path()
    src_depth_image_path = frame_pair_dataset.get_source_depth_image_path()
    tgt_color_image_path = frame_pair_dataset.get_target_color_image_path()
    tgt_depth_image_path = frame_pair_dataset.get_target_depth_image_path()

    # Source color and depth
    source_rgbxyz, _, cropper = DeformDataset.load_image(
        src_color_image_path, src_depth_image_path, intrinsics, image_height, image_width
    )

    # Target color and depth (and boundary mask)
    target_rgbxyz, target_boundary_mask, _ = DeformDataset.load_image(
        tgt_color_image_path, tgt_depth_image_path, intrinsics, image_height, image_width, cropper=cropper,
        max_boundary_dist=max_boundary_distance, compute_boundary_mask=True
    )

    # Graph
    graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = \
        DeformDataset.load_graph_data(frame_pair_dataset.get_sequence_directory(),
                                      frame_pair_dataset.graph_filename, False, cropper)
    pixel_anchors, pixel_weights = DeformDataset.load_anchors_and_weights_from_sequence_directory_and_graph_filename(
        frame_pair_dataset.get_sequence_directory(),
        frame_pair_dataset.graph_filename, cropper)

    num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)

    # Update intrinsics to reflect the crops
    fx, fy, cx, cy = utils.image.modify_intrinsics_due_to_cropping(
        intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
        image_height, image_width, original_h=cropper.h, original_w=cropper.w
    )

    intrinsics = np.zeros((4), dtype=np.float32)
    intrinsics[0] = fx
    intrinsics[1] = fy
    intrinsics[2] = cx
    intrinsics[3] = cy

    #####################################################################################################
    # region ======= Predict deformation ================================================================
    #####################################################################################################

    # Move to device and unsqueeze in the batch dimension (to have batch size 1)
    source_rgbxyz_cuda = torch.from_numpy(source_rgbxyz).cuda().unsqueeze(0)
    target_rgbxyz_cuda = torch.from_numpy(target_rgbxyz).cuda().unsqueeze(0)
    target_boundary_mask_cuda = torch.from_numpy(target_boundary_mask).cuda().unsqueeze(0)
    graph_nodes_cuda = torch.from_numpy(graph_nodes).cuda().unsqueeze(0)
    graph_edges_cuda = torch.from_numpy(graph_edges).cuda().unsqueeze(0)
    graph_edges_weights_cuda = torch.from_numpy(graph_edges_weights).cuda().unsqueeze(0)
    graph_clusters_cuda = torch.from_numpy(graph_clusters).cuda().unsqueeze(0)
    pixel_anchors_cuda = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
    pixel_weights_cuda = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
    intrinsics_cuda = torch.from_numpy(intrinsics).cuda().unsqueeze(0)

    num_nodes_cuda = torch.from_numpy(num_nodes).cuda().unsqueeze(0)

    with torch.no_grad():
        model_data = model(
            source_rgbxyz_cuda, target_rgbxyz_cuda,
            graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
            pixel_anchors_cuda, pixel_weights_cuda,
            num_nodes_cuda, intrinsics_cuda,
            evaluate=True, split="test"
        )

    # Get some of the results
    rotations_pred = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
    translations_pred = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()
    if save_node_transformations:
        # Save rotations & translations
        with open('output/{:s}_{:s}_{:06d}_{:06d}_rotations.np'.format(
                frame_pair_name, segment_name, source_frame_index, target_frame_index), 'wb') as file:
            np.save(file, rotations_pred)
        with open('output/{:s}_{:s}_{:06d}_{:06d}_translations.np'.format(
                frame_pair_name, segment_name, source_frame_index, target_frame_index), 'wb') as file:
            np.save(file, translations_pred)

    mask_pred = model_data["mask_pred"]
    assert mask_pred is not None, "Make sure use_mask=True in options.py"
    mask_pred = mask_pred.view(-1, options.alignment_image_height, options.alignment_image_width).cpu().numpy()

    # Compute mask gt for mask baseline
    _, source_points, valid_source_points, target_matches, valid_target_matches, valid_correspondences, _, _ \
        = model_data["correspondence_info"]

    target_matches = target_matches.view(-1, options.alignment_image_height, options.alignment_image_width).cpu().numpy()
    valid_source_points = valid_source_points.view(-1, options.alignment_image_height, options.alignment_image_width).cpu().numpy()
    valid_target_matches = valid_target_matches.view(-1, options.alignment_image_height, options.alignment_image_width).cpu().numpy()
    valid_correspondences = valid_correspondences.view(-1, options.alignment_image_height, options.alignment_image_width).cpu().numpy()

    # Delete tensors to free up memory
    del source_rgbxyz_cuda
    del target_rgbxyz_cuda
    del target_boundary_mask_cuda
    del graph_nodes_cuda
    del graph_edges_cuda
    del graph_edges_weights_cuda
    del graph_clusters_cuda
    del pixel_anchors_cuda
    del pixel_weights_cuda
    del intrinsics_cuda

    del model

    # endregion

    tracking_viz.visualize_tracking(source_rgbxyz, target_rgbxyz, pixel_anchors, pixel_weights,
                                    graph_nodes, graph_edges, rotations_pred, translations_pred, mask_pred,
                                    valid_source_points, valid_correspondences, target_matches)


if __name__ == "__main__":
    main()
