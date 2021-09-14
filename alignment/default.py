import os
import typing

import open3d.core as o3c
import torch
from alignment.deform_net import DeformNet
from settings import settings_general
from telemetry.telemetry_generator import TelemetryGenerator


def construct_deform_net_from_settings_file(telemetry_generator: typing.Union[TelemetryGenerator, None] = None) -> DeformNet:
    return DeformNet(
        settings_general.alignment_image_width,
        settings_general.alignment_image_height,
        settings_general.skip_solver,
        settings_general.freeze_optical_flow_net,
        settings_general.freeze_mask_net,
        settings_general.gn_depth_sampling_mode,
        settings_general.gn_max_depth,
        settings_general.gn_min_nodes,
        settings_general.gn_max_nodes,
        settings_general.gn_max_matches_train,
        settings_general.gn_max_matches_train_per_batch,
        settings_general.gn_max_matches_eval,
        settings_general.gn_max_warped_points,
        settings_general.gn_debug,
        settings_general.gn_print_timings,
        settings_general.gn_use_edge_weighting,
        settings_general.gn_check_condition_num,
        settings_general.gn_break_on_condition_num,
        settings_general.gn_max_condition_num,
        settings_general.gn_remove_clusters_with_few_matches,
        settings_general.gn_min_num_correspondences_per_cluster,
        settings_general.gn_num_iter,
        settings_general.gn_data_flow,
        settings_general.gn_data_depth,
        settings_general.gn_arap,
        settings_general.gn_lm_factor,
        settings_general.use_mask,
        settings_general.threshold_mask_predictions,
        settings_general.threshold,
        settings_general.patchwise_threshold_mask_predictions,
        settings_general.patch_size,
        settings_general.use_batch_norm,
        telemetry_generator
    )


def load_default_nnrt_network(device_type: o3c.Device.DeviceType, telemetry_generator: typing.Union[TelemetryGenerator, None] = None) -> DeformNet:
    saved_model = settings_general.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct alignment
    if device_type == o3c.Device.CUDA:
        deform_net = construct_deform_net_from_settings_file(telemetry_generator).cuda()
    elif device_type == o3c.Device.CPU:
        deform_net = construct_deform_net_from_settings_file(telemetry_generator)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    if "chairs_things" in saved_model:
        deform_net.flow_net.load_state_dict(pretrained_dict)
    else:
        if settings_general.model_module_to_load == "full_model":
            # Load completely alignment
            deform_net.load_state_dict(pretrained_dict)
        elif settings_general.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = deform_net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            deform_net.load_state_dict(model_dict)
        else:
            print(settings_general.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    deform_net.eval()
    return deform_net
