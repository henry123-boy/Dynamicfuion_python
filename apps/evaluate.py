import os
import json

import numpy as np
from tqdm import tqdm

from alignment.loss import EPE_3D_eval
from data import DeformDataset

from settings import Parameters, process_arguments
import data.io as io


def main():
    #####################################################################################################
    # Options
    #####################################################################################################

    # Parse command line arguments.
    process_arguments()

    split = Parameters.evaluate_split.value.value

    dataset_base_dir = Parameters.path.dataset_base_directory.value
    experiments_dir = Parameters.path.nn_data_directory.value
    model_name = Parameters.model.model_name.value
    gn_max_depth = Parameters.deform_net.gn_max_depth.value

    # Image dimensions to which we crop the input images, such that they are divisible by 64
    alignment_image_height = Parameters.alignment.image_height.value
    alignment_image_width = Parameters.alignment.image_width.value

    #####################################################################################################
    # Read labels and check existence of output dir
    #####################################################################################################

    labels_json = os.path.join(dataset_base_dir, f"{split}_graphs.json")

    assert os.path.isfile(labels_json), f"{labels_json} does not exist! Make sure you specified the correct 'dataset_base_dir' in options.py."

    with open(labels_json, 'r') as f:
        labels = json.loads(f.read())

    # Output dir
    model_base_dir = os.path.join(experiments_dir, "models", model_name)
    predictions_dir = f"{model_base_dir}/evaluation/{split}"
    if not os.path.isdir(predictions_dir):
        raise Exception(f"Predictions directory {predictions_dir} does not exist. Please generate predictions with 'run_generate.sh' first.")

    #####################################################################################################
    # Go over dataset
    #####################################################################################################

    # Graph error (EPE on graph canonical_node_positions)
    graph_error_3d_sum = 0.0
    total_num_nodes = 0

    # Dense EPE 3D
    epe3d_sum = 0.0
    total_num_points = 0

    for label in tqdm(labels):
        assert "graph_node_deformations" in label, "It's highly probable that you're running this script with 'split' set to 'test'. " \
                                                   "but the public dataset does not provide gt for the test set. Plase choose 'val' if " \
                                                   "you want to compute metrics."

        ##############################################################################################
        # Load gt
        ##############################################################################################
        src_color_image_path = os.path.join(dataset_base_dir, label["source_color"])
        src_depth_image_path = os.path.join(dataset_base_dir, label["source_depth"])
        graph_nodes_path = os.path.join(dataset_base_dir, label["graph_nodes"])
        graph_edges_path = os.path.join(dataset_base_dir, label["graph_edges"])
        graph_edges_weights_path = os.path.join(dataset_base_dir, label["graph_edges_weights"])
        graph_node_deformations_path = os.path.join(dataset_base_dir, label["graph_node_deformations"])
        graph_clusters_path = os.path.join(dataset_base_dir, label["graph_clusters"])
        pixel_anchors_path = os.path.join(dataset_base_dir, label["pixel_anchors"])
        pixel_weights_path = os.path.join(dataset_base_dir, label["pixel_weights"])
        optical_flow_image_path = os.path.join(dataset_base_dir, label["optical_flow"])
        scene_flow_image_path = os.path.join(dataset_base_dir, label["scene_flow"])

        intrinsics = label["intrinsics"]

        # Source color and depth
        source, _, cropper = DeformDataset.load_image(
            src_color_image_path, src_depth_image_path, intrinsics, alignment_image_height, alignment_image_width
        )
        source_points = source[3:, :, :]

        # Graph
        graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters = \
            DeformDataset.load_graph_data(
                graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path, graph_clusters_path
            )

        pixel_anchors, pixel_weights = DeformDataset.load_anchors_and_weights(pixel_anchors_path, pixel_weights_path, cropper)

        optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask = DeformDataset.load_flow(
            optical_flow_image_path, scene_flow_image_path, cropper
        )

        # mask is duplicated across feature dimension, so we can safely take the first channel
        scene_flow_mask = scene_flow_mask[0].astype(bool)
        optical_flow_mask = optical_flow_mask[0].astype(bool)

        # All points that have valid optical flow should also have valid scene flow
        assert np.array_equal(scene_flow_mask, optical_flow_mask)

        num_source_points = np.sum(scene_flow_mask)

        # if num_source_points > 100000:
        #     print(label["source_color"], num_source_points)

        ##############################################################################################
        # Load predictions
        ##############################################################################################
        seq_id = label["seq_id"]
        object_id = label["object_id"]
        source_id = label["source_id"]
        target_id = label["target_id"]

        sample_name = f"{seq_id}_{object_id}_{source_id}_{target_id}"

        node_translations_pred_file = os.path.join(predictions_dir, f"{sample_name}_node_translations.bin")
        scene_flow_pred_file = os.path.join(predictions_dir, f"{sample_name}_sceneflow.sflow")

        assert os.path.isfile(
            node_translations_pred_file), f"{node_translations_pred_file} does not exist. Make sure you are not missing any prediction."
        assert os.path.isfile(scene_flow_pred_file), f"{scene_flow_pred_file} does not exist. Make sure you are not missing any prediction."

        node_translations_pred = io.load_graph_nodes_or_deformations(
            node_translations_pred_file
        )

        scene_flow_pred = io.load_flow(
            scene_flow_pred_file
        )

        ##############################################################################################
        # Compute metrics
        ##############################################################################################        

        ######################
        # Node translations (graph_node_deformations are the groundtruth graph canonical_node_positions translations)
        ######################
        graph_error_3d_dict = EPE_3D_eval(
            graph_node_deformations, node_translations_pred
        )
        graph_error_3d_sum += graph_error_3d_dict["sum"]
        total_num_nodes += graph_error_3d_dict["num"]

        ######################
        # Scene flow
        ######################

        # First, get valid source points
        source_anchor_validity = np.all(pixel_anchors >= 0.0, axis=2)

        valid_source_points = np.logical_and.reduce([
            source_points[2, :, :] > 0.0,
            source_points[2, :, :] <= gn_max_depth,
            source_anchor_validity,
            scene_flow_mask,
            optical_flow_mask
        ])

        scene_flow_gt = np.moveaxis(scene_flow_gt, 0, -1)
        scene_flow_pred = np.moveaxis(scene_flow_pred, 0, -1)

        deformed_points_gt = scene_flow_gt[valid_source_points]
        deformed_points_pred = scene_flow_pred[valid_source_points]

        epe_3d_dict = EPE_3D_eval(
            deformed_points_gt, deformed_points_pred
        )
        epe3d_sum += epe_3d_dict["sum"]
        total_num_points += epe_3d_dict["num"]

    # Compute average errors
    graph_error_3d_avg = graph_error_3d_sum / total_num_nodes
    epe3d_avg = epe3d_sum / total_num_points

    print(f"Graph Error 3D (mm): {graph_error_3d_avg * 1000.0}")
    print(f"EPE 3D (mm):         {epe3d_avg * 1000.0}")

    # Write to file
    with open(f"{model_base_dir}/{model_name}__ON__{split}.txt", "w") as f:
        f.write("\n")
        f.write("Evaluation results:\n\n")
        f.write("\n")
        f.write("Model: {0}\n".format(model_name))
        f.write("Split: {0}\n".format(split))
        f.write("\n")
        f.write("{:<40} {}\n".format("Graph Error 3D (mm)", graph_error_3d_avg * 1000.0))
        f.write("{:<40} {}\n".format("EPE 3D (mm)", epe3d_avg * 1000.0))


if __name__ == "__main__":
    main()
