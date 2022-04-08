import os
import typing

import open3d.core as o3c
import torch
from alignment.deform_net import DeformNet
from settings.model import get_saved_model, ModelParameters
from telemetry.telemetry_generator import TelemetryGenerator


def load_default_nnrt_network(device_type: o3c.Device.DeviceType, output_gn_point_clouds=False) -> DeformNet:
    saved_model = get_saved_model()

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct alignment
    if device_type == o3c.Device.CUDA:
        deform_net = DeformNet(output_gn_point_clouds).cuda()
    elif device_type == o3c.Device.CPU:
        deform_net = DeformNet(output_gn_point_clouds)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    if "chairs_things" in saved_model:
        deform_net.flow_net.load_state_dict(pretrained_dict)
    else:
        if ModelParameters.model_module_to_load.value == "full_model":
            # Load completely alignment
            deform_net.load_state_dict(pretrained_dict)
        elif ModelParameters.model_module_to_load.value == "only_flow_net":
            # Load only optical flow part
            model_dict = deform_net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            deform_net.load_state_dict(model_dict)
        else:
            print(ModelParameters.model_module_to_load.value, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    deform_net.eval()
    return deform_net
