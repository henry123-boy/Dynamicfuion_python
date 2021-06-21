import os
import hashlib
import sys
from collections import namedtuple

#####################################################################################################################
# DATA OPTIONS
#####################################################################################################################
# dataset_base_dir    = "/cluster/lothlann/data/nonrigid/public/"
# dataset_base_dir    = "/mnt/slurm_cluster/lothlann/data/nonrigid/hidden/"
# add your mac address & directory to your base dir below
from utils.hardware_id import get_mac_address

LocalPathCollection = namedtuple("LocalPathCollection", "deep_deform_root output")

# to add your own root DeepDeform data directory, run the sha256 cypher on your MAC address and add the hash &
# local directory as a key/value pair to the dict below
workspace = "."
default_output_directory = os.path.join(workspace, "output")
custom_paths_by_mac_address_hash = {
    "1121160f73049dc62efd5cd3ae58daec06185e2d330e680caa46e7f66504f2bf":
        LocalPathCollection(deep_deform_root="/mnt/Data/Reconstruction/real_data/deepdeform",
                            output="/mnt/Data/Reconstruction/output/NerualTracking_experiment_output"),  # ethernet
    "d721c6dceb2f2795bdfc8ff9390adaa3a84ff8f56ddb25a1681b54f5496257e6":
        LocalPathCollection(deep_deform_root="/mnt/Data/Reconstruction/real_data/deepdeform",
                            output="/mnt/Data/Reconstruction/output/NerualTracking_experiment_output"),  # wifi usb card
    "b30b1d8924b5397f69a8367dc4eb4c79e0de8d9000460541c3de0265c63f1414":
        LocalPathCollection(deep_deform_root="/mnt/Data/Datasets/deepdeform",
                            output=default_output_directory),
    "39b8f926e29ac02e90bac6f7d0671f23d8115b4b42457973e38ed871e07b684c":
        LocalPathCollection(deep_deform_root="/mnt/Data/Datasets/deepdeform",
                            output=default_output_directory)
}
try:
    paths = custom_paths_by_mac_address_hash[hashlib.sha256((get_mac_address()).encode('utf-8')).hexdigest()]
    dataset_base_directory = paths.deep_deform_root
    output_directory = paths.output

except KeyError as error:
    raise KeyError(f"Please update the hash at the top of options.py"
                   f" with {hashlib.sha256((get_mac_address()).encode('utf-8')).hexdigest()} as key and "
                   "the tuple containing (path_to_your_DeepDeform_dataset_root, desired_output_directory) as value.") \
        from error

experiments_directory = os.path.join(workspace, "experiments")

alignment_image_width = 640
alignment_image_height = 448
# TODO: depth_scale should be part of dataset & loaded from disk!
depth_scale = 1000
num_worker_threads = 6
num_threads = 4

num_samples_eval = 700
#####################################################################################################################
# GRAPH OPTIONS
#####################################################################################################################
node_coverage = 0.05

#####################################################################################################################
# MODEL INFO
#####################################################################################################################

# Info for a saved alignment
# - In train.py, this info is only used if use_pretrained_model=True
# - In generate.py, evaluate.py or example_viz.py, it is used regardless of the value of use_pretrained_model

use_pretrained_model = False  # used only in train.py

model_module_to_load = "full_model"  # A: "only_flow_net", B: "full_model"
model_name = "model_A"  # your alignment's name
model_iteration = 0  # iteration number of the alignment you want to load

saved_model = os.path.join(experiments_directory, "models", model_name, f"{model_name}_{model_iteration}.pt")

#####################################################################################################################
# TRAINING OPTIONS
#####################################################################################################################
mode = "0_flow"  # ["0_flow", "1_solver", "2_mask", "3_refine"]

if mode == "0_flow":
    from settings.settings_flow import *
elif mode == "1_solver":
    from settings.settings_solver import *
elif mode == "2_mask":
    from settings.settings_mask import *
elif mode == "3_refine":
    from settings.settings_refine import *
elif mode == "4_your_custom_settings":
    # from settings.4_your_custom_settings import *
    pass


#####################################################################################################################
# Print options
#####################################################################################################################

# GPU id
def print_hyperparams():
    print("HYPERPARAMETERS:")
    print()

    print("\tnum_worker_threads           ", num_worker_threads)

    if use_pretrained_model:
        print("\tPretrained alignment              \"{}\"".format(saved_model))
        print("\tModel part to load:          ", model_module_to_load)
        print("\tfreeze_optical_flow_net      ", freeze_optical_flow_net)
        print("\tfreeze_mask_net              ", freeze_mask_net)
    else:
        print("\tPretrained alignment              None")

    print()
    print("\tuse_adam                     ", use_adam)
    print("\tbatch_size                   ", batch_size)
    print("\tevaluation_frequency         ", evaluation_frequency)
    print("\tepochs                       ", epochs)
    print("\tlearning_rate                ", learning_rate)
    if use_lr_scheduler:
        print("\tstep_lr                      ", step_lr)
    else:
        print("\tstep_lr                      ", "None")
    print("\tweight_decay                 ", weight_decay)
    print()
    print("\tgn_max_matches_train         ", gn_max_matches_train)
    print("\tgn_max_matches_eval          ", gn_max_matches_eval)
    print("\tgn_depth_sampling_mode       ", gn_depth_sampling_mode)
    print("\tgn_num_iter                  ", gn_num_iter)
    print("\tgn_data_flow                      ", gn_data_flow)
    print("\tgn_data_depth                      ", gn_data_depth)
    print("\tgn_arap                      ", gn_arap)
    print("\tgn_lm_factor                 ", gn_lm_factor)
    print("\tgn_use_edge_weighting        ", gn_use_edge_weighting)
    print("\tgn_remove_clusters           ", gn_remove_clusters_with_few_matches)
    print()
    print("\tmin_neg_flowed_dist          ", min_neg_flowed_source_to_target_dist)
    print("\tmax_neg_flowed_dist          ", max_pos_flowed_source_to_target_dist)
    print("\tmax_boundary_dist            ", max_boundary_dist)
    print()
    print("\tflow_loss_type               ", flow_loss_type)
    print("\tuse_flow_loss                ", use_flow_loss, "\t", lambda_flow)
    print("\tuse_graph_loss               ", use_graph_loss, "\t", lambda_graph)
    print("\tuse_warp_loss                ", use_warp_loss, "\t", lambda_warp)
    print("\tuse_mask_loss                ", use_mask_loss, "\t", lambda_mask)
    print()
    print("\tuse_mask                     ", use_mask)
