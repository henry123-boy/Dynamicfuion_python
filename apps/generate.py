import os
import json

from alignment.default import load_default_nnrt_network

# 3rd-party
import torch
import numpy as np
from tqdm import tqdm
import open3d.core as o3c

# local

import image_processing
from data import DeformDataset
import data.io as io
from settings import Parameters, process_arguments
from settings.model import get_saved_model


def main():
    #####################################################################################################
    # Options
    #####################################################################################################

    # Parse command line arguments.
    process_arguments()

    split = Parameters.generate_split.value.value  # the second value gets the lc string representation

    # Model checkpoint to use
    saved_model = get_saved_model()
    # Dataset dir
    dataset_base_dir = Parameters.path.dataset_base_directory.value

    # Image dimensions to which we crop the input images, such that they are divisible by 64
    image_height = Parameters.alignment.image_height.value
    image_width = Parameters.alignment.image_width.value

    if Parameters.deform_net.gn_max_matches_eval.value != 100000:
        raise ValueError(f"For whatever sunny reason (based on legacy code), "
                         f"{Parameters.deform_net.gn_max_matches_eval.value} must be exactly 100000")

    if Parameters.deform_net.threshold_mask_predictions.value:
        raise ValueError(f"For whatever sunny reason (based on legacy code), "
                         f"{Parameters.deform_net.threshold_mask_predictions.value} must be set to False for generate.py")

    #####################################################################################################
    # Read labels and assert existence of output dir
    #####################################################################################################

    labels_json = os.path.join(dataset_base_dir, f"{split}_graphs.json")

    assert os.path.isfile(labels_json), f"{labels_json} does not exist! Make sure you specified the correct 'data_root_dir'."

    with open(labels_json, 'r') as f:
        labels = json.loads(f.read())

    # Output dir
    output_dir = os.path.join(Parameters.path.nn_data_directory.value, "models", Parameters.model.model_name.value)
    output_dir = f"{output_dir}/evaluation/{split}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Created output dir", output_dir)
        print()

    #####################################################################################################
    # Model
    #####################################################################################################

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."

    # Construct alignment
    model = load_default_nnrt_network(o3c.Device.DeviceType.CUDA)

    #####################################################################################################
    # Go over dataset
    #####################################################################################################

    for label in tqdm(labels):
        src_color_image_path = os.path.join(dataset_base_dir, label["source_color"])
        src_depth_image_path = os.path.join(dataset_base_dir, label["source_depth"])
        tgt_color_image_path = os.path.join(dataset_base_dir, label["target_color"])
        tgt_depth_image_path = os.path.join(dataset_base_dir, label["target_depth"])
        graph_nodes_path = os.path.join(dataset_base_dir, label["graph_nodes"])
        graph_edges_path = os.path.join(dataset_base_dir, label["graph_edges"])
        graph_edges_weights_path = os.path.join(dataset_base_dir, label["graph_edges_weights"])
        graph_clusters_path = os.path.join(dataset_base_dir, label["graph_clusters"])
        pixel_anchors_path = os.path.join(dataset_base_dir, label["pixel_anchors"])
        pixel_weights_path = os.path.join(dataset_base_dir, label["pixel_weights"])

        intrinsics = label["intrinsics"]

        print(src_color_image_path)

        # Source color and depth
        source, _, cropper = DeformDataset.load_image(
            src_color_image_path, src_depth_image_path, intrinsics, image_height, image_width
        )

        source_points = np.copy(source[3:, :, :])  # 3, h, w

        # Target color and depth (and boundary mask)
        target, _, _ = DeformDataset.load_image(
            tgt_color_image_path, tgt_depth_image_path, intrinsics, image_height, image_width, cropper=cropper,
            max_boundary_distance=None, compute_boundary_mask=False
        )

        # Graph
        graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters = \
            DeformDataset.load_graph_data(
                graph_nodes_path, graph_edges_path, graph_edges_weights_path, None, graph_clusters_path
            )
        pixel_anchors, pixel_weights = DeformDataset.load_anchors_and_weights(pixel_anchors_path, pixel_weights_path, cropper)

        num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)

        # Update intrinsics to reflect the crops
        fx, fy, cx, cy = image_processing.modify_intrinsics_due_to_cropping(
            intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
            image_height, image_width, original_h=cropper.h, original_w=cropper.w
        )

        intrinsics = np.zeros((4), dtype=np.float32)
        intrinsics[0] = fx
        intrinsics[1] = fy
        intrinsics[2] = cx
        intrinsics[3] = cy

        #####################################################################################################
        # Predict deformation
        #####################################################################################################

        # Move to device and unsqueeze in the batch dimension (to have batch size 1)
        source_cuda = torch.from_numpy(source).cuda().unsqueeze(0)
        target_cuda = torch.from_numpy(target).cuda().unsqueeze(0)
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
                source_cuda, target_cuda,
                graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                pixel_anchors_cuda, pixel_weights_cuda,
                num_nodes_cuda, intrinsics_cuda,
                evaluate=True, split="test"
            )

        # Get predicted graph deformation
        node_rotations_pred = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
        node_translations_pred = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()

        # Warp source points with predicted graph deformation
        warped_source_points = image_processing.warp_deform_3d(
            source, pixel_anchors, pixel_weights, graph_nodes, node_rotations_pred, node_translations_pred
        )

        # Compute dense 3d flow
        scene_flow_pred = warped_source_points - source_points

        # Save predictions
        seq_id = label["seq_id"]
        object_id = label["object_id"]
        source_id = label["source_id"]
        target_id = label["target_id"]

        sample_name = f"{seq_id}_{object_id}_{source_id}_{target_id}"

        node_translations_pred_file = os.path.join(output_dir, f"{sample_name}_node_translations.bin")
        scene_flow_pred_file = os.path.join(output_dir, f"{sample_name}_sceneflow.sflow")

        io.save_graph_node_deformations(
            node_translations_pred_file, node_translations_pred
        )

        io.save_flow(
            scene_flow_pred_file, scene_flow_pred
        )


if __name__ == "__main__":
    main()
