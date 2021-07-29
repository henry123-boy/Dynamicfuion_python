import os
import torch
from alignment.deform_net import DeformNet
from settings import settings_general
from telemetry.telemetry_generator import TelemetryGenerator


def load_default_nnrt_network(telemetry_generator: TelemetryGenerator) -> DeformNet:
    saved_model = settings_general.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct alignment
    deform_net = DeformNet(telemetry_generator).cuda()

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
